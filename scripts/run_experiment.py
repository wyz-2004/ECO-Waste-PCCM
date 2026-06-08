from __future__ import annotations

import json
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
if os.environ.get("ECOWASTE_PROJECT_ROOT"):
    ROOT = Path(os.environ["ECOWASTE_PROJECT_ROOT"]).resolve()
elif SCRIPT_PATH.parent.name == "scripts" and SCRIPT_PATH.parent.parent.name == "code":
    ROOT = SCRIPT_PATH.parents[2]
else:
    ROOT = SCRIPT_PATH.parents[1]

if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import ecowaste_core as base


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PROTOCOL_DIR = OUT_DIR / "protocol"

SEEDS = [11, 23, 37, 53, 71]
ECO_METHOD = "ECO-Waste-PCCM"
EPS = 1e-8


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, FIG_DIR, PROTOCOL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def finite_col_stat(values: np.ndarray, mask: np.ndarray, stat: str = "median") -> np.ndarray:
    arr = np.where(mask, values, np.nan)
    with np.errstate(all="ignore"):
        if stat == "mean":
            out = np.nanmean(arr, axis=0)
        else:
            out = np.nanmedian(arr, axis=0)
    fallback = np.nanmedian(values, axis=0)
    fallback = np.where(np.isfinite(fallback), fallback, 0.0)
    return np.where(np.isfinite(out), out, fallback)


def stat_imputer(data: base.PreparedData, train_mask: np.ndarray, stat: str) -> np.ndarray:
    col = finite_col_stat(data.model_values, train_mask, stat)
    return np.tile(col, (data.model_values.shape[0], 1))


def softimpute(
    data: base.PreparedData,
    train_mask: np.ndarray,
    rank: int = 18,
    shrink: float = 0.18,
    epochs: int = 32,
) -> np.ndarray:
    base_pred = base.fill_group_median(data, train_mask)
    X = data.model_values
    filled = np.where(train_mask, X, base_pred)
    for _ in range(epochs):
        col_center = filled.mean(axis=0)
        centered = filled - col_center
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            s2 = np.maximum(s - shrink, 0.0)
            r = min(rank, int(np.sum(s2 > 1e-10)))
            if r == 0:
                recon = np.tile(col_center, (filled.shape[0], 1))
            else:
                recon = (u[:, :r] * s2[:r]) @ vt[:r, :] + col_center
        except np.linalg.LinAlgError:
            recon = filled
        filled = np.where(train_mask, X, 0.82 * recon + 0.18 * base_pred)
        filled = np.clip(filled, -8, 8)
    return np.where(np.isfinite(filled), filled, base_pred)


def ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lam: float = 1.0) -> np.ndarray:
    if len(y_train) < 4 or X_train.shape[1] == 0:
        return np.full(X_test.shape[0], float(np.nanmedian(y_train)) if len(y_train) else 0.0)
    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0) + 1e-6
    mu_y = float(y_train.mean())
    xs = (X_train - mu_x) / sd_x
    xt = (X_test - mu_x) / sd_x
    ys = y_train - mu_y
    gram = xs.T @ xs
    gram.flat[:: gram.shape[0] + 1] += lam
    try:
        coef = np.linalg.solve(gram, xs.T @ ys)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ xs.T @ ys
    return xt @ coef + mu_y


def choose_predictors(train_mask: np.ndarray, target: int, max_features: int = 42) -> np.ndarray:
    obs = train_mask.mean(axis=0)
    candidates = np.where(obs >= 0.08)[0]
    candidates = candidates[candidates != target]
    if len(candidates) <= max_features:
        return candidates
    order = np.argsort(obs[candidates])[::-1]
    return candidates[order[:max_features]]


def mice_ridge_imputer(
    data: base.PreparedData,
    train_mask: np.ndarray,
    iterations: int = 4,
    max_features: int = 42,
) -> np.ndarray:
    X = data.model_values
    filled = np.where(train_mask, X, base.fill_group_median(data, train_mask))
    n_cols = X.shape[1]
    update_order = np.argsort(train_mask.mean(axis=0))
    for _ in range(iterations):
        for j in update_order:
            obs = train_mask[:, j]
            miss = ~obs
            if obs.sum() < 12 or not miss.any():
                continue
            predictors = choose_predictors(train_mask, j, max_features)
            if len(predictors) == 0:
                continue
            y = X[obs, j]
            pred = ridge_predict(filled[obs][:, predictors], y, filled[miss][:, predictors], lam=2.5)
            filled[miss, j] = 0.72 * filled[miss, j] + 0.28 * pred
        filled = np.where(train_mask, X, np.clip(filled, -8, 8))
    return np.where(np.isfinite(filled), filled, base.fill_group_median(data, train_mask))


def tree_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    rng: np.random.Generator,
    depth: int = 3,
    min_leaf: int = 8,
    max_features: int = 18,
) -> np.ndarray:
    def build(idx: np.ndarray, d: int) -> dict:
        y = y_train[idx]
        value = float(y.mean()) if len(y) else 0.0
        if d == 0 or len(idx) < 2 * min_leaf or float(np.var(y)) < 1e-8:
            return {"value": value}
        p = X_train.shape[1]
        feats = rng.choice(p, size=min(max_features, p), replace=False)
        best = None
        best_loss = np.inf
        for f in feats:
            vals = X_train[idx, f]
            qs = np.unique(np.quantile(vals, [0.25, 0.5, 0.75]))
            for thr in qs:
                left = idx[vals <= thr]
                right = idx[vals > thr]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                loss = float(np.var(y_train[left]) * len(left) + np.var(y_train[right]) * len(right))
                if loss < best_loss:
                    best_loss = loss
                    best = (f, float(thr), left, right)
        if best is None:
            return {"value": value}
        f, thr, left, right = best
        return {"value": value, "feature": f, "threshold": thr, "left": build(left, d - 1), "right": build(right, d - 1)}

    def apply_one(node: dict, row: np.ndarray) -> float:
        while "feature" in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["value"])

    root = build(np.arange(X_train.shape[0]), depth)
    return np.array([apply_one(root, row) for row in X_test], dtype=float)


def missforest_imputer(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    iterations: int = 2,
    trees: int = 5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = data.model_values
    group = base.fill_group_median(data, train_mask)
    filled = np.where(train_mask, X, group)
    update_order = np.argsort(train_mask.mean(axis=0))
    for _ in range(iterations):
        for j in update_order:
            obs = train_mask[:, j]
            miss = ~obs
            if obs.sum() < 18 or not miss.any():
                continue
            predictors = choose_predictors(train_mask, j, max_features=36)
            if len(predictors) < 2:
                continue
            preds = []
            for _tree in range(trees):
                boot = rng.choice(np.where(obs)[0], size=int(obs.sum()), replace=True)
                sub = rng.choice(predictors, size=min(len(predictors), 24), replace=False)
                pred = tree_predict(filled[boot][:, sub], X[boot, j], filled[miss][:, sub], rng)
                preds.append(pred)
            filled[miss, j] = 0.65 * filled[miss, j] + 0.35 * np.mean(preds, axis=0)
        filled = np.where(train_mask, X, np.clip(filled, -8, 8))
    return np.where(np.isfinite(filled), filled, group)


def denoising_autoencoder_imputer(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    hidden: int = 44,
    epochs: int = 80,
    lr: float = 0.010,
    noise: float = 0.18,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = data.model_values
    group = base.fill_group_median(data, train_mask)
    filled = np.where(train_mask, X, group)
    n, p = filled.shape
    h = min(hidden, max(8, p // 3))
    w1 = rng.normal(0, 0.08, size=(p, h))
    b1 = np.zeros(h)
    w2 = rng.normal(0, 0.08, size=(h, p))
    b2 = np.zeros(p)
    obs_weight = train_mask.astype(float)
    obs_weight = obs_weight / max(obs_weight.mean(), 1e-3)
    for _ in range(epochs):
        drop = rng.random(filled.shape) < noise
        x_in = np.where(drop & ~train_mask, 0.0, filled)
        z = np.tanh(x_in @ w1 + b1)
        out = z @ w2 + b2
        grad = (out - X) * obs_weight / n
        grad = np.where(train_mask, grad, 0.0)
        gw2 = z.T @ grad
        gb2 = grad.sum(axis=0)
        gz = grad @ w2.T * (1.0 - z**2)
        gw1 = x_in.T @ gz
        gb1 = gz.sum(axis=0)
        w2 -= lr * (gw2 + 1e-4 * w2)
        b2 -= lr * gb2
        w1 -= lr * (gw1 + 1e-4 * w1)
        b1 -= lr * gb1
        if _ % 8 == 7:
            recon = np.tanh(filled @ w1 + b1) @ w2 + b2
            filled = np.where(train_mask, X, 0.82 * filled + 0.18 * recon)
            filled = np.clip(filled, -8, 8)
    recon = np.tanh(filled @ w1 + b1) @ w2 + b2
    return np.where(train_mask, X, 0.65 * recon + 0.35 * group)


def gain_style_imputer(data: base.PreparedData, train_mask: np.ndarray, seed: int) -> np.ndarray:
    # A lightweight GAIN-style baseline: denoising reconstruction plus hint-mask
    # reliability. It is intentionally reported as "GAIN-style" because the
    # local runtime has no deep-learning framework.
    dae = denoising_autoencoder_imputer(data, train_mask, seed=seed + 100, hidden=36, epochs=60, noise=0.28)
    graph = base.weighted_city_knn(data, train_mask, k=10)
    hint = train_mask.mean(axis=0)
    blend = 0.45 + 0.35 * hint
    return blend[None, :] * dae + (1.0 - blend[None, :]) * graph


def pccm_predict(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    use_evidence: bool = True,
    use_graph: bool = True,
    use_constraints: bool = True,
    use_conformal_ready_uncertainty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    original_reliability = data.reliability.copy()
    if not use_evidence:
        data.reliability = np.full_like(data.reliability, 0.55, dtype=float)
    try:
        group = base.fill_group_median(data, train_mask)
        graph = base.weighted_city_knn(data, train_mask, k=14) if use_graph else group
        low_rank = base.matrix_factorization(data, train_mask, rank=14, epochs=36, seed=seed, use_evidence=use_evidence)
        soft = softimpute(data, train_mask, rank=20, shrink=0.12, epochs=24)
        rel = np.clip(np.nan_to_num(data.reliability, nan=0.45), 0.18, 1.0)
        graph_gate = np.clip(0.25 + 0.35 * train_mask.mean(axis=1), 0.20, 0.62)
        feature_missing = 1.0 - train_mask.mean(axis=0)
        sparse_gate = np.clip(feature_missing, 0.18, 0.72)
        core = 0.48 * low_rank + 0.22 * soft + 0.30 * graph
        smoothed = graph_gate[:, None] * graph + (1.0 - graph_gate[:, None]) * core
        pred = (1.0 - sparse_gate)[None, :] * smoothed + sparse_gate[None, :] * group
        if use_constraints:
            pred = base.project_percent_constraints(data, pred)
        disagreement = np.std(np.stack([group, graph, low_rank, soft], axis=0), axis=0)
        rel_gap = 1.0 - np.where(rel > 0, rel, 0.45)
        uncertainty = 0.40 * disagreement + 0.32 * feature_missing[None, :] + 0.28 * rel_gap
        if not use_conformal_ready_uncertainty:
            uncertainty = 0.6 * uncertainty + 0.4 * disagreement
        return np.where(np.isfinite(pred), pred, group), np.maximum(uncertainty, 1e-4)
    finally:
        data.reliability = original_reliability


def scenario_masks(data: base.PreparedData, seed: int) -> dict[str, np.ndarray]:
    scenarios = {
        "random_20pct": base.make_random_mask(data, 0.20, seed),
        "block_by_system": base.make_block_mask(data, seed + 3),
    }
    if "region_id" in data.context.columns:
        scenarios["region_holdout"] = base.make_group_holdout_mask(data, "region_id", seed + 5)
    if "income_id_2022" in data.context.columns:
        scenarios["income_holdout"] = base.make_group_holdout_mask(data, "income_id_2022", seed + 7)
    return scenarios


def method_registry(seed: int) -> dict[str, Callable[[base.PreparedData, np.ndarray], tuple[np.ndarray, np.ndarray | None]]]:
    return {
        "mean": lambda d, m: (stat_imputer(d, m, "mean"), None),
        "median": lambda d, m: (stat_imputer(d, m, "median"), None),
        "region_income_median": lambda d, m: (base.fill_group_median(d, m), None),
        "knn_imputer": lambda d, m: (base.weighted_city_knn(d, m, k=10), None),
        "mice_ridge": lambda d, m: (mice_ridge_imputer(d, m), None),
        "missforest": lambda d, m: (missforest_imputer(d, m, seed=seed), None),
        "softimpute": lambda d, m: (softimpute(d, m, rank=18, shrink=0.16), None),
        "nuclear_norm_mc": lambda d, m: (softimpute(d, m, rank=24, shrink=0.30), None),
        "low_rank_svd": lambda d, m: (base.matrix_factorization(d, m, rank=14, epochs=34, seed=seed, use_evidence=False), None),
        "dae": lambda d, m: (denoising_autoencoder_imputer(d, m, seed=seed), None),
        "gain_style": lambda d, m: (gain_style_imputer(d, m, seed=seed), None),
        ECO_METHOD: lambda d, m: pccm_predict(d, m, seed=seed),
    }


def raw_prediction(data: base.PreparedData, pred: np.ndarray) -> pd.DataFrame:
    raw = pred * data.feature_scale + data.feature_mean
    return pd.DataFrame(raw, columns=data.features)


def constraint_metrics(data: base.PreparedData, pred: np.ndarray) -> dict[str, float]:
    raw = raw_prediction(data, pred)
    idx = {f: j for j, f in enumerate(data.features)}
    out = base.constraint_violation(data, pred)
    pct_cols = [f for f in data.features if f in data.percent_features]
    if pct_cols:
        pct = raw[pct_cols].to_numpy(dtype=float)
        low = np.maximum(0.0, -pct)
        high = np.maximum(0.0, pct - 1.0)
        out["range_violation_rate"] = float(np.mean((low + high) > 1e-9))
        out["range_violation_magnitude"] = float(np.mean(low + high))
    else:
        out["range_violation_rate"] = np.nan
        out["range_violation_magnitude"] = np.nan
    needed = [
        "msw_total_msw_generated_tons_per_year",
        "population_number_of_people",
        "msw_total_msw_generated_kg_per_cap_per_day",
    ]
    if all(f in idx for f in needed):
        tons = np.expm1(np.clip(raw[needed[0]].to_numpy(dtype=float), 0, 30))
        pop = np.expm1(np.clip(raw[needed[1]].to_numpy(dtype=float), 0, 30))
        kg = raw[needed[2]].to_numpy(dtype=float)
        implied = pop * kg * 365.0 / 1000.0
        valid = np.isfinite(tons) & np.isfinite(implied) & (tons > 1) & (implied > 1)
        out["mass_balance_relative_error"] = float(np.median(np.abs(tons[valid] - implied[valid]) / np.maximum(tons[valid], 1.0))) if valid.any() else np.nan
    else:
        out["mass_balance_relative_error"] = np.nan
    return out


def evaluate_group(data: base.PreparedData, pred: np.ndarray, test_mask: np.ndarray, group: str) -> dict[str, float]:
    cols = np.array([base.group_id(f) == group for f in data.features])
    mask = test_mask & cols[None, :]
    metrics = base.evaluate(pred, data.model_values, mask, None)
    metrics["percent_mae_points"] = base.percent_mae_pp(data, pred, mask)
    return metrics


def aggregate_metrics(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c not in set(keys)
        and c != "seed"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df.groupby(keys, dropna=False)
    mean = grouped[metric_cols].mean().reset_index()
    std = grouped[metric_cols].std(ddof=1).reset_index()
    std = std.rename(columns={c: f"{c}_std" for c in metric_cols})
    out = mean.merge(std, on=keys, how="left")
    out["seeds"] = grouped.size().to_numpy()
    return out


def run_imputation_suite_v4(
    data: base.PreparedData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    group_rows = []
    final_preds: dict[str, np.ndarray] = {}
    final_unc: dict[str, np.ndarray] = {}
    groups = sorted({base.group_id(f) for f in data.features})
    for seed in SEEDS:
        for scenario, test_mask in scenario_masks(data, seed).items():
            train_mask = data.observed_mask & ~test_mask
            for method, fn in method_registry(seed).items():
                pred, unc = fn(data, train_mask)
                pred = np.where(np.isfinite(pred), pred, base.fill_group_median(data, train_mask))
                metrics = base.evaluate(pred, data.model_values, test_mask, unc)
                metrics.update(constraint_metrics(data, pred))
                metrics["percent_mae_points"] = base.percent_mae_pp(data, pred, test_mask)
                metrics.update({"scenario": scenario, "method": method, "seed": seed})
                rows.append(metrics)
                for group in groups:
                    gm = evaluate_group(data, pred, test_mask, group)
                    gm.update({"scenario": scenario, "method": method, "seed": seed, "variable_group": group})
                    group_rows.append(gm)
                if seed == SEEDS[0] and scenario == "random_20pct":
                    final_preds[method] = pred
                    final_unc[method] = unc if unc is not None else np.zeros_like(pred)
    raw = pd.DataFrame(rows)
    summary = aggregate_metrics(raw, ["scenario", "method"])
    group_raw = pd.DataFrame(group_rows)
    return raw, summary, group_raw, final_preds, final_unc


def run_ablation_suite_v4(data: base.PreparedData) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        test_mask = base.make_random_mask(data, 0.20, seed + 101)
        train_mask = data.observed_mask & ~test_mask
        variants = {
            "full_ECO-Waste-PCCM": dict(),
            "fixed_expert_mixture": dict(use_adaptive_gate=False),
            "global_learned_gate": dict(gate_mode="global"),
            "group_only_gate": dict(gate_mode="group"),
            "reliability_only_gate": dict(gate_mode="reliability"),
            "static_rule_evidence": dict(evidence_source="static"),
            "no_codebook_evidence": dict(use_evidence=False),
            "no_city_graph": dict(use_graph=False),
            "no_feasibility_projection": dict(use_constraints=False),
            "proxy_uncertainty_only": dict(use_conformal_ready_uncertainty=False),
        }
        for variant, kwargs in variants.items():
            evidence_source = kwargs.pop("evidence_source", "learned")
            if evidence_source == "static":
                original = data.reliability.copy()
                data.reliability = data.static_reliability.copy()
                try:
                    pred, unc = pccm_predict(data, train_mask, seed=seed, **kwargs)
                finally:
                    data.reliability = original
            else:
                pred, unc = pccm_predict(data, train_mask, seed=seed, **kwargs)
            metrics = base.evaluate(pred, data.model_values, test_mask, unc)
            metrics.update(constraint_metrics(data, pred))
            metrics["percent_mae_points"] = base.percent_mae_pp(data, pred, test_mask)
            metrics.update({"variant": variant, "scenario": "random_20pct_ablation", "seed": seed})
            rows.append(metrics)
    return aggregate_metrics(pd.DataFrame(rows), ["scenario", "variant"])


def write_split_and_hyperparameter_files(data: base.PreparedData) -> None:
    split_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for scenario, test_mask in scenario_masks(data, seed).items():
            for city_idx, feature_idx in np.argwhere(test_mask):
                split_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "city_code": data.city.iloc[int(city_idx)]["city_code"],
                        "feature": data.features[int(feature_idx)],
                        "assignment": "heldout_test",
                    }
                )
    pd.DataFrame(split_rows).to_csv(PROTOCOL_DIR / "masking_split_assignments.csv", index=False, encoding="utf-8-sig")
    hyperparameters = [
        ("seeds", str(SEEDS), "All main masking protocols"),
        ("gate_validation_rate", "0.09; fallback 0.16", "Self-supervised gate validation cells"),
        ("gate_modes", "fixed, global, group, reliability, group_reliability", "Gate hierarchy ablation"),
        ("gate_prior", "[0.28, 0.24, 0.30, 0.18]", "Median, graph, weighted low rank, SoftImpute"),
        ("gate_ridge", "0.10 / 0.16 / 0.22", "Global / marginal / full-context gates"),
        ("gate_optimizer", "projected gradient, max 220 iterations", "Nonnegative sum-to-one weights"),
        ("city_graph_k", "14", "City-graph expert"),
        ("weighted_low_rank", "rank=14, epochs=34", "Reliability-weighted factorization"),
        ("softimpute", "rank=20, shrink=0.12, epochs=22", "SoftImpute expert"),
        ("projection", "[0,1] box plus exact unit simplex", "Percentage and flow feasibility"),
        ("acquisition_budgets", "{0,5,10,20,40,80,120}", "Budget frontier"),
    ]
    pd.DataFrame(hyperparameters, columns=["parameter", "value", "role"]).to_csv(
        PROTOCOL_DIR / "hyperparameters.csv", index=False, encoding="utf-8-sig"
    )


def conformal_from_masks(
    data: base.PreparedData,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    calib_mask: np.ndarray,
    test_mask: np.ndarray,
    reliability_weighted: bool,
) -> dict[str, float]:
    calib_err = np.abs(pred[calib_mask] - data.model_values[calib_mask])
    test_err = np.abs(pred[test_mask] - data.model_values[test_mask])
    calib_unc = np.maximum(uncertainty[calib_mask], 1e-4)
    test_unc = np.maximum(uncertainty[test_mask], 1e-4)
    if reliability_weighted:
        calib_rel = np.clip(data.reliability[calib_mask], 0.2, 1.0)
        test_rel = np.clip(data.reliability[test_mask], 0.2, 1.0)
        calib_score = calib_err / (calib_unc / np.sqrt(calib_rel))
        test_scale = test_unc / np.sqrt(test_rel)
    else:
        calib_score = calib_err / calib_unc
        test_scale = test_unc
    out: dict[str, float] = {}
    ece_terms = []
    for target in [0.50, 0.80, 0.90]:
        if calib_score.size == 0 or test_err.size == 0:
            cov = width = np.nan
        else:
            q = float(np.quantile(calib_score, min(0.995, target)))
            width_vec = 2.0 * q * test_scale
            cov = float(np.mean(test_err <= width_vec / 2.0))
            width = float(np.mean(width_vec))
        out[f"coverage_{int(target * 100)}"] = cov
        out[f"width_{int(target * 100)}"] = width
        if np.isfinite(cov):
            ece_terms.append(abs(cov - target))
    out["ece"] = float(np.mean(ece_terms)) if ece_terms else np.nan
    out["crps_proxy"] = float(np.mean(np.minimum(test_err, np.quantile(test_err, 0.95)))) if test_err.size else np.nan
    return out


def run_conformal_calibration(data: base.PreparedData) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 200)
        test_mask = base.make_random_mask(data, 0.20, seed + 201)
        pool = data.observed_mask & ~test_mask
        calib = pool & (rng.random(pool.shape) < 0.15)
        train = pool & ~calib
        methods = {
            "SVD + naive interval": lambda: (
                base.matrix_factorization(data, train, rank=14, epochs=34, seed=seed, use_evidence=False),
                np.tile(np.nanstd(np.where(train, data.model_values, np.nan), axis=0), (data.model_values.shape[0], 1)),
                False,
            ),
            "ECO proxy uncertainty": lambda: (*pccm_predict(data, train, seed=seed), False),
            "ECO + conformal calibration": lambda: (*pccm_predict(data, train, seed=seed), True),
        }
        for method, fn in methods.items():
            pred, unc, weighted = fn()
            unc = np.where(np.isfinite(unc), unc, np.nanmedian(np.abs(pred - np.nanmedian(pred, axis=0))))
            met = conformal_from_masks(data, pred, unc, calib, test_mask, reliability_weighted=weighted)
            met.update({"method": method, "seed": seed, "calibration_cells": int(calib.sum()), "test_cells": int(test_mask.sum())})
            rows.append(met)
    return aggregate_metrics(pd.DataFrame(rows), ["method"])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    order = np.argsort(scores)[::-1]
    y = labels[order]
    if y.sum() == 0:
        return np.nan
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision[y]) / y.sum())


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = labels[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels.astype(float))[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else np.nan


def risk_score_array(data: base.PreparedData, pred: np.ndarray) -> np.ndarray:
    risk = base.risk_components(data, pred, np.zeros_like(pred))
    return risk.sort_values("city_code")["risk_score"].to_numpy(dtype=float)


def run_risk_identification(data: base.PreparedData, final_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    full_reference = base.fill_group_median(data, data.observed_mask)
    truth_scores = risk_score_array(data, full_reference)
    labels = truth_scores >= np.quantile(truth_scores, 0.75)
    rows = []
    for method in ["median", "softimpute", "low_rank_svd", "missforest", ECO_METHOD]:
        if method not in final_preds:
            continue
        scores = risk_score_array(data, final_preds[method])
        pred_labels = scores >= np.quantile(scores, 0.75)
        rows.append(
            {
                "method": method,
                "auroc": auroc(scores, labels),
                "auprc": auprc(scores, labels),
                "macro_f1": macro_f1(labels, pred_labels),
                "ndcg_at_20": ndcg_at_k(scores, labels, 20),
                "precision_at_20": float(labels[np.argsort(scores)[::-1][:20]].mean()),
            }
        )
    return pd.DataFrame(rows)


def macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for cls in [False, True]:
        tp = np.sum((pred == cls) & (labels == cls))
        fp = np.sum((pred == cls) & (labels != cls))
        fn = np.sum((pred != cls) & (labels == cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        vals.append(2 * precision * recall / max(precision + recall, EPS))
    return float(np.mean(vals))


def policy_metrics(peer_effects: pd.DataFrame, recs: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    if peer_effects.empty:
        return pd.DataFrame([{"metric": "policy_hit_at_3", "value": np.nan}])
    positive = peer_effects["estimated_risk_reduction"] > 0
    top3 = peer_effects.sort_values("estimated_risk_reduction", ascending=False).head(3)
    risk_iqr = float(risk["risk_score"].quantile(0.75) - risk["risk_score"].quantile(0.25))
    metrics = [
        {"metric": "Policy Hit@3", "value": float((top3["estimated_risk_reduction"] > 0).mean()), "description": "Top peer-matched policy groups with positive risk gap"},
        {"metric": "Peer Match Agreement", "value": float(positive.mean()), "description": "Share of region/income peer comparisons with lower risk among treated peers"},
        {"metric": "Risk-gap Explanation Score", "value": float(np.maximum(peer_effects["estimated_risk_reduction"], 0).mean() / max(risk_iqr, EPS)), "description": "Positive peer risk gap normalized by risk IQR"},
        {"metric": "Actionable Recommendations", "value": float(len(recs)), "description": "Number of high-risk city action rows passing peer-match filters"},
    ]
    return pd.DataFrame(metrics)


def active_acquisition_v4(data: base.PreparedData) -> pd.DataFrame:
    risk_vars = [
        "waste_treatment_open_dumpsite_percent",
        "waste_uncollected_percent",
        "waste_treatment_unaccounted_for_percent",
        "waste_treatment_recycling_percent",
        "waste_treatment_compost_percent",
        "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
        "waste_collection_coverage_total_percent_of_population",
        "waste_collection_coverage_total_percent_of_waste",
    ]
    cols = [data.features.index(f) for f in risk_vars if f in data.features]
    rng = np.random.default_rng(902)
    eligible = data.observed_mask.copy()
    colmask = np.zeros(eligible.shape[1], dtype=bool)
    colmask[cols] = True
    heldout = eligible & colmask[None, :] & (rng.random(eligible.shape) < 0.42)
    entries = np.argwhere(heldout)
    if len(entries) < 30:
        return pd.DataFrame()
    budgets = [0, 5, 10, 20, 40, 80, 120]
    base_train = data.observed_mask & ~heldout
    base_pred, base_unc = pccm_predict(data, base_train, seed=777)
    risk = base.risk_components(data, base_pred, base_unc).set_index("city_code")
    city_risk = risk.loc[data.city["city_code"], "risk_percentile"].to_numpy(dtype=float)
    var_missing = 1.0 - base_train.mean(axis=0)
    var_importance = np.array([1.0 if j in cols else 0.35 for j in range(len(data.features))])
    evidence_gap = 1.0 - np.clip(np.nan_to_num(data.reliability, nan=0.35), 0.0, 1.0)
    constraint_group = np.array([1.0 if base.group_id(f) in {"composition", "treatment", "collection"} else 0.35 for f in data.features])
    strategy_scores = {
        "Random": rng.random(len(entries)),
        "Missingness-only": np.array([var_missing[j] for _, j in entries]),
        "Uncertainty-only": np.array([base_unc[i, j] for i, j in entries]),
        "Risk-only": np.array([city_risk[i] * var_importance[j] for i, j in entries]),
        "Constraint-only": np.array([constraint_group[j] for _, j in entries]),
        "ECO-Acquire": np.array(
            [
                0.18 * base_unc[i, j]
                + 0.12 * city_risk[i]
                + 0.46 * var_missing[j]
                + 0.12 * var_importance[j]
                + 0.08 * evidence_gap[i, j]
                + 0.04 * constraint_group[j]
                for i, j in entries
            ]
        ),
        "ECO-Acquire-no-evidence": np.array(
            [
                0.20 * base_unc[i, j]
                + 0.13 * city_risk[i]
                + 0.51 * var_missing[j]
                + 0.12 * var_importance[j]
                + 0.04 * constraint_group[j]
                for i, j in entries
            ]
        ),
    }
    full_train = data.observed_mask.copy()
    full_pred, _ = pccm_predict(data, full_train, seed=778)
    ref_rank = rankdata(risk_score_array(data, full_pred))
    rows = []
    for strategy, scores in strategy_scores.items():
        order = np.argsort(scores)[::-1]
        for budget in budgets:
            reveal = entries[order[: min(budget, len(entries))]]
            train = base_train.copy()
            if len(reveal):
                train[reveal[:, 0], reveal[:, 1]] = True
            remaining = heldout & ~train
            pred, unc = pccm_predict(data, train, seed=800 + budget)
            metrics = base.evaluate(pred, data.model_values, remaining, unc)
            metrics["percent_mae_points"] = base.percent_mae_pp(data, pred, remaining)
            cons = constraint_metrics(data, pred)
            pred_rank = rankdata(risk_score_array(data, pred))
            risk_rank_corr = float(np.corrcoef(ref_rank, pred_rank)[0, 1]) if len(ref_rank) > 2 else np.nan
            rows.append(
                {
                    **metrics,
                    "strategy": strategy,
                    "budget_cells": budget,
                    "remaining_test_cells": int(remaining.sum()),
                    "composition_sum_abs_error": cons.get("composition_sum_abs_error", np.nan),
                    "treatment_sum_abs_error": cons.get("treatment_sum_abs_error", np.nan),
                    "range_violation_rate": cons.get("range_violation_rate", np.nan),
                    "mass_balance_relative_error": cons.get("mass_balance_relative_error", np.nan),
                    "risk_rank_correlation": risk_rank_corr,
                }
            )
    return pd.DataFrame(rows)


def active_acquisition_summary(acquisition: pd.DataFrame) -> pd.DataFrame:
    if acquisition.empty:
        return pd.DataFrame()
    base_rows = acquisition[acquisition["budget_cells"] == 0]
    base_mae = float(base_rows["mae_norm"].mean()) if not base_rows.empty else np.nan
    rows = []
    for budget, grp in acquisition.groupby("budget_cells"):
        best = grp.sort_values(["mae_norm", "risk_rank_correlation"], ascending=[True, False]).iloc[0]
        eco = grp[grp["strategy"] == "ECO-Acquire"].iloc[0] if (grp["strategy"] == "ECO-Acquire").any() else best
        rows.append(
            {
                "budget_cells": int(budget),
                "best_strategy_by_mae": best["strategy"],
                "best_mae_norm": float(best["mae_norm"]),
                "best_risk_rank_correlation": float(best["risk_rank_correlation"]),
                "eco_acquire_mae_norm": float(eco["mae_norm"]),
                "eco_acquire_improvement_vs_no_acquisition": base_mae - float(eco["mae_norm"]),
                "eco_acquire_risk_rank_correlation": float(eco["risk_rank_correlation"]),
            }
        )
    return pd.DataFrame(rows).sort_values("budget_cells")


def write_v4_report(
    data: base.PreparedData,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    conformal: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    policy_metric_df: pd.DataFrame,
    acquisition_summary_df: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, rows: int = 12) -> str:
        if df.empty:
            return "_No rows._"
        view = df.head(rows).copy()
        cols = list(view.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in view.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append("" if not np.isfinite(val) else f"{val:.4f}")
                else:
                    vals.append(str(val).replace("|", "/"))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep, *body])

    lines = [
        "# ECO-Waste-PCCM v4 Experiment Report",
        "",
        "This run follows the final experimental plan: the method is framed as provenance-calibrated constrained matrix completion, evidence is used in reconstruction/calibration/acquisition, and evaluation expands beyond raw imputation error.",
        "",
        "## Dataset",
        f"- Cities: {len(data.city)}",
        f"- Raw city fields: {len(data.city.columns)}",
        f"- Modeled features: {len(data.features)}",
        f"- Observed model-matrix rate: {data.observed_mask.mean():.2%}",
        "",
        "## Multi-seed imputation benchmark",
        md_table(metrics.sort_values(["scenario", "rmse_norm"]).head(24), 24),
        "",
        "## PCCM ablation",
        md_table(ablation.sort_values("mae_norm")),
        "",
        "## Conformal uncertainty calibration",
        md_table(conformal),
        "",
        "## High-risk city identification",
        md_table(risk_metrics),
        "",
        "## Observational peer-matched policy prioritization",
        md_table(policy_metric_df),
        "",
        "## Active acquisition",
        md_table(acquisition_summary_df),
        "",
        "## Important wording constraints",
        "- Use `observational peer-matched policy prioritization`; do not claim causal intervention effects.",
        "- State that ECO-Waste-PCCM is accuracy-competitive with strong baselines and strongest on feasibility/calibration/acquisition, unless a table directly supports a stronger accuracy claim.",
        "- State that GAIN-style and MissForest implementations are lightweight local baselines implemented without external data or unavailable deep-learning/scikit-learn runtimes.",
    ]
    (OUT_DIR / "ECO-Waste-PCCM_v4_report.md").write_text("\n".join(lines), encoding="utf-8")


GATING_RECORDS: list[dict[str, object]] = []


def simplex_projection(vector: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    values = np.asarray(vector, dtype=float)
    if values.size == 0:
        return values.copy()
    ordered = np.sort(values)[::-1]
    cssv = np.cumsum(ordered) - total
    rho_candidates = ordered - cssv / np.arange(1, len(values) + 1) > 0
    if not rho_candidates.any():
        return np.full_like(values, total / len(values))
    rho = int(np.where(rho_candidates)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(values - theta, 0.0)


def weighted_simplex_projection(vector: np.ndarray, weights: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Weighted Euclidean projection aligned with normalized evaluation scale."""
    values = np.asarray(vector, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 1e-4, 1e4)
    if values.size == 0:
        return values.copy()

    def projected_sum(lam: float) -> float:
        return float(np.maximum(values - lam / w, 0.0).sum())

    lo, hi = -1.0, 1.0
    while projected_sum(lo) < total:
        lo *= 2.0
    while projected_sum(hi) > total:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if projected_sum(mid) > total:
            lo = mid
        else:
            hi = mid
    out = np.maximum(values - hi / w, 0.0)
    return out * (total / max(out.sum(), 1e-12))


def exact_feasibility_projection(data: base.PreparedData, pred_norm: np.ndarray) -> np.ndarray:
    """Project percentages and disjoint flow groups onto closed convex feasible sets."""
    raw = pred_norm * data.feature_scale + data.feature_mean
    feature_index = {f: j for j, f in enumerate(data.features)}
    for feature in data.percent_features:
        if feature in feature_index:
            raw[:, feature_index[feature]] = np.clip(raw[:, feature_index[feature]], 0.0, 1.0)
    groups = [
        [f for f in data.features if f.startswith("composition_msw_") and f.endswith("_percent")],
        [
            f
            for f in data.features
            if (f.startswith("waste_treatment_") and f.endswith("_percent")) or f == "waste_uncollected_percent"
        ],
    ]
    for features in groups:
        idx = [feature_index[f] for f in features if f in feature_index]
        if len(idx) < 3:
            continue
        normalized_weights = 1.0 / np.maximum(data.feature_scale[idx], 1e-4) ** 2
        normalized_weights = normalized_weights / max(float(np.median(normalized_weights)), 1e-8)
        raw[:, idx] = np.vstack(
            [weighted_simplex_projection(row, normalized_weights, 1.0) for row in raw[:, idx]]
        )
    return (raw - data.feature_mean) / data.feature_scale


def unweighted_feasibility_projection(data: base.PreparedData, pred_norm: np.ndarray) -> np.ndarray:
    """Projection ablation using ordinary Euclidean simplex geometry."""
    raw = pred_norm * data.feature_scale + data.feature_mean
    feature_index = {f: j for j, f in enumerate(data.features)}
    for feature in data.percent_features:
        if feature in feature_index:
            raw[:, feature_index[feature]] = np.clip(raw[:, feature_index[feature]], 0.0, 1.0)
    groups = [
        [f for f in data.features if f.startswith("composition_msw_") and f.endswith("_percent")],
        [
            f
            for f in data.features
            if (f.startswith("waste_treatment_") and f.endswith("_percent")) or f == "waste_uncollected_percent"
        ],
    ]
    for features in groups:
        idx = [feature_index[f] for f in features if f in feature_index]
        if len(idx) >= 3:
            raw[:, idx] = np.vstack([simplex_projection(row, 1.0) for row in raw[:, idx]])
    return (raw - data.feature_mean) / data.feature_scale


def convex_weights(expert_matrix: np.ndarray, target: np.ndarray, prior: np.ndarray | None = None, ridge: float = 0.08) -> np.ndarray:
    """Fit nonnegative sum-to-one expert weights with projected gradient."""
    A = np.nan_to_num(np.asarray(expert_matrix, dtype=float), nan=0.0)
    y = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0)
    k = A.shape[1]
    prior = np.full(k, 1.0 / k) if prior is None else simplex_projection(prior)
    if len(y) < max(20, 4 * k):
        return prior
    w = prior.copy()
    spectral = float(np.linalg.norm(A, ord=2) ** 2 / max(len(y), 1))
    step = 0.65 / max(2.0 * spectral + 2.0 * ridge, 1e-6)
    for _ in range(220):
        grad = 2.0 * (A.T @ (A @ w - y)) / len(y) + 2.0 * ridge * (w - prior)
        updated = simplex_projection(w - step * grad)
        if np.linalg.norm(updated - w) < 1e-8:
            w = updated
            break
        w = updated
    return w


def reliability_bins(reliability: np.ndarray) -> np.ndarray:
    return np.digitize(np.nan_to_num(reliability, nan=0.45), [0.42, 0.62], right=True)


def build_experts(
    data: base.PreparedData,
    mask: np.ndarray,
    seed: int,
    use_evidence: bool,
    use_graph: bool,
    use_low_rank: bool = True,
    use_softimpute: bool = True,
) -> tuple[list[str], np.ndarray]:
    group = base.fill_group_median(data, mask)
    graph = base.weighted_city_knn(data, mask, k=14) if use_graph else group
    low_rank = (
        base.matrix_factorization(data, mask, rank=14, epochs=34, seed=seed, use_evidence=use_evidence)
        if use_low_rank
        else group
    )
    soft = softimpute(data, mask, rank=20, shrink=0.12, epochs=22) if use_softimpute else group
    return ["group_median", "city_graph", "weighted_low_rank", "softimpute"], np.stack([group, graph, low_rank, soft], axis=0)


def learned_gate(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    use_evidence: bool,
    use_graph: bool,
    gate_mode: str = "group_reliability",
    use_low_rank: bool = True,
    use_softimpute: bool = True,
    use_cross_fitting: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, int], np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed + 601)
    eligible = train_mask.copy()
    eligible[:, train_mask.sum(axis=0) < 18] = False
    validation = eligible & (rng.random(eligible.shape) < 0.09)
    if validation.sum() < 220:
        validation = eligible & (rng.random(eligible.shape) < 0.16)
    fit_mask = train_mask & ~validation if use_cross_fitting else train_mask
    names, fit_experts = build_experts(
        data, fit_mask, seed + 7, use_evidence, use_graph, use_low_rank, use_softimpute
    )
    A = fit_experts[:, validation].T
    y = data.model_values[validation]
    prior = np.array([0.28, 0.24, 0.30, 0.18])
    global_w = convex_weights(A, y, prior=prior, ridge=0.10)
    rel_bins = reliability_bins(data.reliability if use_evidence else np.full_like(data.reliability, 0.55))
    groups = np.array([base.group_id(f) for f in data.features])
    reliability_weights = {
        rel_bin: convex_weights(
            fit_experts[:, validation & (rel_bins == rel_bin)].T,
            data.model_values[validation & (rel_bins == rel_bin)],
            prior=global_w,
            ridge=0.16,
        )
        for rel_bin in range(3)
    }
    context_weights: dict[tuple[str, int], np.ndarray] = {}
    residual_scale = np.full((len(data.city), len(data.features)), np.nan, dtype=float)
    global_residual = np.abs(A @ global_w - y)
    global_scale = float(np.nanmedian(global_residual)) if global_residual.size else 0.4
    for group in sorted(set(groups)):
        col_mask = groups == group
        group_val = validation & col_mask[None, :]
        group_w = convex_weights(fit_experts[:, group_val].T, data.model_values[group_val], prior=global_w, ridge=0.16)
        for rel_bin in range(3):
            context = group_val & (rel_bins == rel_bin)
            if gate_mode == "global":
                w = global_w
            elif gate_mode == "group":
                w = group_w
            elif gate_mode == "reliability":
                w = reliability_weights[rel_bin]
            elif gate_mode == "group_reliability":
                w = convex_weights(fit_experts[:, context].T, data.model_values[context], prior=group_w, ridge=0.22)
            else:
                raise ValueError(f"Unknown gate mode: {gate_mode}")
            context_weights[(group, rel_bin)] = w
            if context.sum() >= 12:
                err = np.abs(fit_experts[:, context].T @ w - data.model_values[context])
                scale = float(np.nanmedian(err))
            else:
                scale = global_scale
            residual_scale[:, col_mask] = np.where((rel_bins[:, col_mask] == rel_bin), scale, residual_scale[:, col_mask])
    residual_scale = np.where(np.isfinite(residual_scale), residual_scale, global_scale)
    _, final_experts = build_experts(
        data, train_mask, seed + 19, use_evidence, use_graph, use_low_rank, use_softimpute
    )
    pred = np.zeros_like(data.model_values)
    for j, group in enumerate(groups):
        for rel_bin in range(3):
            rows = rel_bins[:, j] == rel_bin
            w = context_weights.get((group, rel_bin), global_w)
            pred[rows, j] = final_experts[:, rows, j].T @ w
    disagreement = np.std(final_experts, axis=0)
    for group in sorted(set(groups)):
        for rel_bin in range(3):
            w = context_weights.get((group, rel_bin), global_w)
            GATING_RECORDS.append(
                {
                    "seed": seed,
                    "variable_group": group,
                    "reliability_bin": ["low", "medium", "high"][rel_bin],
                    "uses_learned_evidence": use_evidence,
                    "uses_city_graph": use_graph,
                    "gate_mode": gate_mode,
                    **{f"weight_{name}": float(value) for name, value in zip(names, w)},
                    "validation_cells": int((validation & (groups == group)[None, :] & (rel_bins == rel_bin)).sum()),
                }
            )
    return pred, disagreement, context_weights, residual_scale


def pccm_predict(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    use_evidence: bool = True,
    use_graph: bool = True,
    use_constraints: bool = True,
    use_conformal_ready_uncertainty: bool = True,
    use_adaptive_gate: bool = True,
    gate_mode: str = "group_reliability",
    use_low_rank: bool = True,
    use_softimpute: bool = True,
    use_cross_fitting: bool = True,
    projection_mode: str = "weighted",
) -> tuple[np.ndarray, np.ndarray]:
    original_reliability = data.reliability.copy()
    if not use_evidence:
        data.reliability = np.full_like(data.reliability, 0.55, dtype=float)
    try:
        group = base.fill_group_median(data, train_mask)
        if use_adaptive_gate:
            pred, disagreement, _, residual_scale = learned_gate(
                data,
                train_mask,
                seed,
                use_evidence,
                use_graph,
                gate_mode=gate_mode,
                use_low_rank=use_low_rank,
                use_softimpute=use_softimpute,
                use_cross_fitting=use_cross_fitting,
            )
        else:
            _, experts = build_experts(
                data, train_mask, seed, use_evidence, use_graph, use_low_rank, use_softimpute
            )
            pred = np.tensordot(np.array([0.28, 0.24, 0.30, 0.18]), experts, axes=([0], [0]))
            disagreement = np.std(experts, axis=0)
            residual_scale = np.full_like(pred, np.nanmedian(disagreement))
        feature_missing = 1.0 - train_mask.mean(axis=0)
        sparse_gate = np.clip((feature_missing - 0.70) / 0.28, 0.0, 0.72)
        pred = (1.0 - sparse_gate)[None, :] * pred + sparse_gate[None, :] * group
        if use_constraints and projection_mode == "weighted":
            pred = exact_feasibility_projection(data, pred)
        elif use_constraints and projection_mode == "unweighted":
            pred = unweighted_feasibility_projection(data, pred)
        rel = np.clip(np.nan_to_num(data.reliability, nan=0.45), 0.12, 1.0)
        uncertainty = 0.38 * disagreement + 0.38 * residual_scale + 0.14 * feature_missing[None, :] + 0.10 * (1.0 - rel)
        if not use_conformal_ready_uncertainty:
            uncertainty = 0.72 * disagreement + 0.28 * feature_missing[None, :]
        return np.where(np.isfinite(pred), pred, group), np.maximum(uncertainty, 1e-4)
    finally:
        data.reliability = original_reliability


def robust_convex_weights(
    expert_matrix: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray | None = None,
    ridge: float = 0.08,
    huber_delta: float = 0.55,
) -> np.ndarray:
    """Fit a convex stack under a smooth MAE/RMSE compromise."""
    A = np.nan_to_num(np.asarray(expert_matrix, dtype=float), nan=0.0)
    y = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0)
    k = A.shape[1]
    prior = np.full(k, 1.0 / k) if prior is None else simplex_projection(prior)
    if len(y) < max(24, 5 * k):
        return prior
    w = prior.copy()
    spectral = float(np.linalg.norm(A, ord=2) ** 2 / max(len(y), 1))
    step = 0.35 / max(spectral + ridge, 1e-6)
    for _ in range(450):
        residual = A @ w - y
        smooth_l1_grad = residual / np.sqrt(residual**2 + huber_delta**2)
        squared_grad = residual
        grad = A.T @ (0.55 * smooth_l1_grad + 0.45 * squared_grad) / len(y)
        grad += ridge * (w - prior)
        updated = simplex_projection(w - step * grad)
        if np.linalg.norm(updated - w) < 1e-9:
            w = updated
            break
        w = updated
    return w


def _safe_stack_candidates(
    data: base.PreparedData,
    mask: np.ndarray,
    seed: int,
    use_evidence: bool,
    use_graph: bool = True,
    use_low_rank: bool = True,
    use_softimpute: bool = True,
    use_cross_fitting: bool = True,
) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    median = stat_imputer(data, mask, "median")
    soft = softimpute(data, mask, rank=20, shrink=0.12, epochs=26) if use_softimpute else median
    svd = (
        base.matrix_factorization(data, mask, rank=14, epochs=36, seed=seed, use_evidence=False)
        if use_low_rank
        else median
    )
    base_pccm, base_unc = pccm_predict(
        data,
        mask,
        seed=seed + 13,
        # Point completion is deliberately evidence-neutral. Evidence is used
        # only by the validation-gated uncertainty adapter below.
        use_evidence=False,
        use_graph=use_graph,
        use_constraints=False,
        gate_mode="group",
        use_low_rank=use_low_rank,
        use_softimpute=use_softimpute,
        use_cross_fitting=use_cross_fitting,
    )
    candidates = np.stack([median, soft, svd, base_pccm], axis=0)
    generic = np.std(candidates, axis=0)
    return ["median", "softimpute", "svd", "base_pccm"], candidates, [generic, base_unc]


def safe_pccm_predict(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    use_evidence: bool = True,
    use_constraints: bool = True,
    use_graph: bool = True,
    use_low_rank: bool = True,
    use_softimpute: bool = True,
    use_cross_fitting: bool = True,
    use_adaptive_stack: bool = True,
    projection_mode: str = "weighted",
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted robust stack with validation-gated evidence adaptation.

    The safe adapter nests the no-evidence solution: evidence affects the
    uncertainty scale only when it improves a held-out validation objective.
    """
    rng = np.random.default_rng(seed + 3301)
    eligible = train_mask.copy()
    eligible[:, train_mask.sum(axis=0) < 24] = False
    validation = eligible & (rng.random(eligible.shape) < 0.13)
    if validation.sum() < 300:
        validation = eligible & (rng.random(eligible.shape) < 0.19)
    fit_mask = train_mask & ~validation if use_cross_fitting else train_mask

    names, fit_candidates, fit_unc_parts = _safe_stack_candidates(
        data,
        fit_mask,
        seed + 7,
        use_evidence,
        use_graph,
        use_low_rank,
        use_softimpute,
        use_cross_fitting,
    )
    groups = np.array([base.group_id(feature) for feature in data.features])
    prior = np.array([0.30, 0.28, 0.18, 0.24])
    global_w = (
        robust_convex_weights(fit_candidates[:, validation].T, data.model_values[validation], prior=prior)
        if use_adaptive_stack
        else np.full(len(names), 1.0 / len(names))
    )
    group_weights: dict[str, np.ndarray] = {}
    feature_bias = np.zeros(len(data.features), dtype=float)
    residual_scale = np.full(len(data.features), np.nan, dtype=float)
    for group in sorted(set(groups)):
        group_validation = validation & (groups == group)[None, :]
        group_weights[group] = (
            robust_convex_weights(
                fit_candidates[:, group_validation].T,
                data.model_values[group_validation],
                prior=global_w,
                ridge=0.12,
            )
            if use_adaptive_stack
            else global_w
        )
    for j, group in enumerate(groups):
        cell_mask = validation[:, j]
        w = group_weights[group]
        if cell_mask.sum() >= 10:
            residual = data.model_values[cell_mask, j] - fit_candidates[:, cell_mask, j].T @ w
            # Bias correction is deliberately shrunk because workbook columns are small.
            feature_bias[j] = 0.55 * float(np.median(residual))
            residual_scale[j] = float(np.median(np.abs(residual - np.median(residual))))
    fallback_scale = float(np.nanmedian(residual_scale[np.isfinite(residual_scale)]))
    residual_scale = np.where(np.isfinite(residual_scale) & (residual_scale > 1e-4), residual_scale, max(fallback_scale, 0.08))

    _, final_candidates, final_unc_parts = _safe_stack_candidates(
        data,
        train_mask,
        seed + 29,
        use_evidence,
        use_graph,
        use_low_rank,
        use_softimpute,
        use_cross_fitting,
    )
    pred = np.zeros_like(data.model_values)
    for j, group in enumerate(groups):
        pred[:, j] = final_candidates[:, :, j].T @ group_weights[group] + feature_bias[j]

    disagreement = np.std(final_candidates, axis=0)
    missingness = 1.0 - train_mask.mean(axis=0)
    base_unc = (
        0.44 * disagreement
        + 0.34 * residual_scale[None, :]
        + 0.14 * missingness[None, :]
        + 0.08 * np.nan_to_num(final_unc_parts[1], nan=0.0)
    )
    selected_beta = 0.0
    if use_evidence and validation.any():
        evidence_gap = 1.0 - np.clip(np.nan_to_num(data.reliability, nan=0.55), 0.08, 1.0)
        centered_gap = evidence_gap - float(np.mean(evidence_gap[validation]))
        validation_error = np.abs(
            fit_candidates[:, validation].T @ global_w - data.model_values[validation]
        )
        validation_base = np.maximum(
            0.55 * fit_unc_parts[0][validation] + 0.45 * np.nan_to_num(fit_unc_parts[1][validation], nan=0.0),
            1e-4,
        )
        best_loss = float(np.mean(validation_error / validation_base + np.log(validation_base)))
        for beta in [-0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00, 1.25]:
            scale = np.maximum(validation_base * np.exp(beta * centered_gap[validation]), 1e-4)
            loss = float(np.mean(validation_error / scale + np.log(scale)))
            if loss < best_loss - 1e-4:
                best_loss = loss
                selected_beta = beta
        base_unc = base_unc * np.exp(selected_beta * centered_gap)

    if use_constraints and projection_mode == "weighted":
        pred = exact_feasibility_projection(data, pred)
    elif use_constraints and projection_mode == "unweighted":
        pred = unweighted_feasibility_projection(data, pred)
    pred = np.where(np.isfinite(pred), pred, stat_imputer(data, train_mask, "median"))
    return pred, np.maximum(np.nan_to_num(base_unc, nan=1.0), 1e-4)


def pccm_l2_predict(
    data: base.PreparedData,
    train_mask: np.ndarray,
    seed: int,
    use_constraints: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Squared-risk head with a validation-selected PCCM/SoftImpute blend."""
    rng = np.random.default_rng(seed + 4401)
    eligible = train_mask.copy()
    eligible[:, train_mask.sum(axis=0) < 24] = False
    validation = eligible & (rng.random(eligible.shape) < 0.16)
    if validation.sum() < 300:
        validation = eligible & (rng.random(eligible.shape) < 0.22)
    fit_mask = train_mask & ~validation
    _, fit_candidates, _ = _safe_stack_candidates(data, fit_mask, seed + 11, use_evidence=False)
    groups = np.array([base.group_id(feature) for feature in data.features])
    prior = np.array([0.10, 0.42, 0.24, 0.24])
    global_w = convex_weights(fit_candidates[:, validation].T, data.model_values[validation], prior=prior, ridge=0.07)
    weights: dict[str, np.ndarray] = {}
    for group in sorted(set(groups)):
        mask = validation & (groups == group)[None, :]
        weights[group] = convex_weights(
            fit_candidates[:, mask].T,
            data.model_values[mask],
            prior=global_w,
            ridge=0.11,
        )

    _, select_candidates, _ = _safe_stack_candidates(data, fit_mask, seed + 31, use_evidence=False)
    stack_fit = np.zeros_like(data.model_values)
    for j, group in enumerate(groups):
        stack_fit[:, j] = select_candidates[:, :, j].T @ weights[group]
    soft_fit = softimpute(data, fit_mask, rank=30, shrink=0.18, epochs=40)

    best_alpha = 1.0
    best_mse = np.inf
    for alpha in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]:
        candidate = alpha * stack_fit + (1.0 - alpha) * soft_fit
        candidate = exact_feasibility_projection(data, candidate)
        mse = float(np.mean((candidate[validation] - data.model_values[validation]) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    _, candidates, unc_parts = _safe_stack_candidates(data, train_mask, seed + 31, use_evidence=False)
    stack_pred = np.zeros_like(data.model_values)
    for j, group in enumerate(groups):
        stack_pred[:, j] = candidates[:, :, j].T @ weights[group]
    soft_pred = softimpute(data, train_mask, rank=30, shrink=0.18, epochs=40)
    pred = best_alpha * stack_pred + (1.0 - best_alpha) * soft_pred
    uncertainty = (
        0.52 * np.std(candidates, axis=0)
        + 0.30 * np.nan_to_num(unc_parts[1], nan=0.0)
        + 0.18 * np.abs(stack_pred - soft_pred)
    )
    if use_constraints:
        pred = exact_feasibility_projection(data, pred)
    return np.where(np.isfinite(pred), pred, stat_imputer(data, train_mask, "median")), np.maximum(uncertainty, 1e-4)


def run_evidence_utility_analysis(data: base.PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 1200)
        test = base.make_random_mask(data, 0.20, seed + 1201)
        pool = data.observed_mask & ~test
        calib = pool & (rng.random(pool.shape) < 0.15)
        train = pool & ~calib
        variants = ["no_evidence", "static_evidence", "learned_evidence"]
        for variant in variants:
            original = data.reliability.copy()
            if variant == "no_evidence":
                data.reliability = np.full_like(data.reliability, 0.55)
                use_evidence = False
                weighted = False
            elif variant == "static_evidence":
                data.reliability = data.static_reliability.copy()
                use_evidence = True
                weighted = True
            else:
                use_evidence = True
                weighted = True
            try:
                pred, unc = pccm_predict(data, train, seed + 30, use_evidence=use_evidence)
                metrics = conformal_from_masks(data, pred, unc, calib, test, reliability_weighted=weighted)
                err = np.abs(pred[test] - data.model_values[test])
                rel = data.reliability[test]
                metrics.update(
                    {
                        "variant": variant,
                        "seed": seed,
                        "mae_norm": float(err.mean()),
                        "reliability_error_spearman": safe_spearman(rel, -err),
                        "test_cells": int(test.sum()),
                    }
                )
                rows.append(metrics)
                if variant == "learned_evidence":
                    quantiles = np.quantile(rel, [0.0, 0.25, 0.50, 0.75, 1.0])
                    for bin_id in range(4):
                        include = (rel >= quantiles[bin_id]) & (rel <= quantiles[bin_id + 1] if bin_id == 3 else rel < quantiles[bin_id + 1])
                        bin_rows.append(
                            {
                                "seed": seed,
                                "reliability_quartile": bin_id + 1,
                                "mean_reliability": float(rel[include].mean()),
                                "mean_absolute_error": float(err[include].mean()),
                                "mean_uncertainty": float(unc[test][include].mean()),
                                "cells": int(include.sum()),
                            }
                        )
            finally:
                data.reliability = original
    return aggregate_metrics(pd.DataFrame(rows), ["variant"]), aggregate_metrics(pd.DataFrame(bin_rows), ["reliability_quartile"])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 4:
        return np.nan
    return float(np.corrcoef(rankdata(a[valid]), rankdata(b[valid]))[0, 1])


RISK_FEATURES = [
    "waste_treatment_open_dumpsite_percent",
    "waste_uncollected_percent",
    "waste_treatment_unaccounted_for_percent",
    "waste_collection_coverage_total_percent_of_population",
    "waste_collection_coverage_total_percent_of_waste",
    "waste_treatment_recycling_percent",
    "waste_treatment_compost_percent",
    "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
]


def run_leave_one_risk_component_out(data: base.PreparedData) -> pd.DataFrame:
    reference = base.fill_group_median(data, data.observed_mask)
    reference_rank = rankdata(risk_score_array(data, reference))
    reference_score = risk_score_array(data, reference)
    labels = reference_score >= np.quantile(reference_score, 0.75)
    rows = []
    for seed in SEEDS:
        for feature in RISK_FEATURES:
            if feature not in data.features:
                continue
            j = data.features.index(feature)
            test = np.zeros_like(data.observed_mask)
            test[:, j] = data.observed_mask[:, j]
            train = data.observed_mask & ~test
            pred, _ = pccm_predict(data, train, seed + j)
            scores = risk_score_array(data, pred)
            rows.append(
                {
                    "seed": seed,
                    "heldout_risk_component": feature,
                    "observed_test_cells": int(test.sum()),
                    "component_mae_norm": float(np.mean(np.abs(pred[test] - data.model_values[test]))) if test.any() else np.nan,
                    "risk_rank_spearman": safe_spearman(reference_rank, rankdata(scores)),
                    "top20_overlap": float(len(set(np.argsort(reference_score)[-20:]) & set(np.argsort(scores)[-20:])) / 20.0),
                    "ndcg_at_20": ndcg_at_k(scores, labels, 20),
                }
            )
    return aggregate_metrics(pd.DataFrame(rows), ["heldout_risk_component"])


def risk_definition_scores(data: base.PreparedData, pred: np.ndarray) -> dict[str, np.ndarray]:
    raw = pred * data.feature_scale + data.feature_mean
    values = pd.DataFrame(raw, columns=data.features)
    get = lambda name: values[name].to_numpy(float) if name in values else np.zeros(len(values))
    coverage = np.maximum(get("waste_collection_coverage_total_percent_of_population"), get("waste_collection_coverage_total_percent_of_waste"))
    bad = get("waste_treatment_open_dumpsite_percent") + get("waste_uncollected_percent") + get("waste_treatment_unaccounted_for_percent")
    good = get("waste_treatment_recycling_percent") + get("waste_treatment_compost_percent") + get("waste_treatment_sanitary_landfill_landfill_gas_system_percent")
    return {
        "balanced": bad + 0.45 * (1.0 - coverage) - good,
        "service_gap": bad + (1.0 - coverage),
        "treatment_hazard": 1.5 * get("waste_treatment_open_dumpsite_percent") + get("waste_uncollected_percent") - 0.5 * good,
        "conservative_bad_flows": bad,
    }


def run_risk_definition_robustness(data: base.PreparedData, final_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    reference = base.fill_group_median(data, data.observed_mask)
    reference_defs = risk_definition_scores(data, reference)
    rows = []
    for method in ["median", "softimpute", "low_rank_svd", ECO_METHOD]:
        if method not in final_preds:
            continue
        definitions = risk_definition_scores(data, final_preds[method])
        for name, truth in reference_defs.items():
            score = definitions[name]
            rows.append(
                {
                    "method": method,
                    "risk_definition": name,
                    "rank_spearman": safe_spearman(truth, score),
                    "top20_overlap": float(len(set(np.argsort(truth)[-20:]) & set(np.argsort(score)[-20:])) / 20.0),
                }
            )
    return pd.DataFrame(rows)


def paired_significance_tests(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eco = raw_metrics[raw_metrics["method"] == ECO_METHOD]
    for scenario in sorted(raw_metrics["scenario"].unique()):
        for baseline in ["median", "softimpute", "low_rank_svd", "missforest"]:
            for metric in ["mae_norm", "rmse_norm", "composition_sum_abs_error", "treatment_sum_abs_error"]:
                left = eco[eco["scenario"] == scenario][["seed", metric]]
                right = raw_metrics[(raw_metrics["scenario"] == scenario) & (raw_metrics["method"] == baseline)][["seed", metric]]
                paired = left.merge(right, on="seed", suffixes=("_eco", "_baseline")).dropna()
                if paired.empty:
                    continue
                diff = paired[f"{metric}_eco"].to_numpy(float) - paired[f"{metric}_baseline"].to_numpy(float)
                observed = abs(float(diff.mean()))
                permutations = [abs(float(np.mean(diff * np.array(signs)))) for signs in itertools.product([-1.0, 1.0], repeat=len(diff))]
                p_value = float(np.mean(np.array(permutations) >= observed - 1e-12))
                rows.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "metric": metric,
                        "eco_minus_baseline": float(diff.mean()),
                        "paired_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                        "exact_sign_flip_p": p_value,
                        "eco_better_direction": bool(diff.mean() < 0),
                        "pairs": len(diff),
                    }
                )
    return pd.DataFrame(rows)


def failure_analysis(group_raw: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_metrics(group_raw, ["scenario", "method", "variable_group"])
    rows = []
    for scenario in sorted(summary["scenario"].unique()):
        eco = summary[(summary["scenario"] == scenario) & (summary["method"] == ECO_METHOD)].set_index("variable_group")
        for baseline in ["softimpute", "low_rank_svd", "median"]:
            other = summary[(summary["scenario"] == scenario) & (summary["method"] == baseline)].set_index("variable_group")
            common = eco.index.intersection(other.index)
            for group in common:
                if not np.isfinite(eco.loc[group, "mae_norm"]) or not np.isfinite(other.loc[group, "mae_norm"]):
                    continue
                delta = float(eco.loc[group, "mae_norm"] - other.loc[group, "mae_norm"])
                if delta > 0:
                    rows.append(
                        {
                            "scenario": scenario,
                            "variable_group": group,
                            "stronger_baseline": baseline,
                            "pccm_mae_norm": float(eco.loc[group, "mae_norm"]),
                            "baseline_mae_norm": float(other.loc[group, "mae_norm"]),
                            "pccm_minus_baseline_mae": delta,
                            "heldout_cells": float(eco.loc[group, "heldout_cells"]),
                            "interpretation": "accuracy cost accepted only when feasibility, calibration, or auditability adds value",
                        }
                    )
    return pd.DataFrame(rows).sort_values("pccm_minus_baseline_mae", ascending=False)


def projection_property_audit(data: base.PreparedData, seed: int = 2027) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, size=data.model_values.shape)
    y = rng.normal(0.0, 2.0, size=data.model_values.shape)
    px = exact_feasibility_projection(data, x)
    py = exact_feasibility_projection(data, y)
    ppx = exact_feasibility_projection(data, px)
    metrics = constraint_metrics(data, px)
    return pd.DataFrame(
        [
            {
                "property": "percentage_range",
                "value": metrics["range_violation_rate"],
                "threshold": 1e-12,
                "passed": metrics["range_violation_rate"] <= 1e-12,
            },
            {
                "property": "composition_simplex_closure",
                "value": metrics["composition_sum_abs_error"],
                "threshold": 1e-12,
                "passed": metrics["composition_sum_abs_error"] <= 1e-12,
            },
            {
                "property": "treatment_simplex_closure",
                "value": metrics["treatment_sum_abs_error"],
                "threshold": 1e-12,
                "passed": metrics["treatment_sum_abs_error"] <= 1e-12,
            },
            {
                "property": "projection_idempotence_max_abs",
                "value": float(np.max(np.abs(ppx - px))),
                "threshold": 1e-10,
                "passed": float(np.max(np.abs(ppx - px))) <= 1e-10,
            },
            {
                "property": "nonexpansive_ratio",
                "value": float(np.linalg.norm(px - py) / max(np.linalg.norm(x - y), EPS)),
                "threshold": 1.0 + 1e-10,
                "passed": float(np.linalg.norm(px - py) / max(np.linalg.norm(x - y), EPS)) <= 1.0 + 1e-10,
            },
        ]
    )


def policy_sensitivity_analysis(peer_effects: pd.DataFrame) -> pd.DataFrame:
    if peer_effects.empty or "estimated_risk_reduction" not in peer_effects:
        return pd.DataFrame()
    ordered = peer_effects.sort_values("estimated_risk_reduction", ascending=False)
    rows = []
    for top_k in [3, 5, 10, 20, len(ordered)]:
        view = ordered.head(min(top_k, len(ordered)))
        rows.append(
            {
                "top_k": min(top_k, len(ordered)),
                "positive_gap_rate": float((view["estimated_risk_reduction"] > 0).mean()),
                "mean_estimated_risk_reduction": float(view["estimated_risk_reduction"].mean()),
                "median_estimated_risk_reduction": float(view["estimated_risk_reduction"].median()),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("top_k")


def write_reproducibility_details(data: base.PreparedData) -> None:
    details = {
        "algorithm": "ECO-Waste-PCCM with hierarchical gate ablation and exact simplex projection",
        "seeds": SEEDS,
        "masking_protocols": ["random_20pct", "block_by_system", "region_holdout", "income_holdout"],
        "gate_validation_rate": 0.09,
        "gate_experts": ["group_median", "city_graph", "weighted_low_rank", "softimpute"],
        "gate_constraints": "nonnegative weights summing to one; group- and reliability-bin-conditioned",
        "gate_ablation": ["fixed", "global", "group_only", "reliability_only", "group_x_reliability"],
        "projection": "Euclidean projection onto [0,1] percentage box and exact composition/treatment simplices",
        "conformal": "separate calibration cells; absolute residual divided by uncertainty and reliability-adjusted scale",
        "modeled_features": len(data.features),
        "cities": len(data.city),
        "source_data": "What a Waste 3.0 City Dataset and Codebook only",
    }
    (PROTOCOL_DIR / "reproducibility_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Reproducibility Details", ""] + [f"- **{key}**: {value}" for key, value in details.items()]
    (PROTOCOL_DIR / "reproducibility_details.md").write_text("\n".join(lines), encoding="utf-8")


def write_experiment_report(
    data: base.PreparedData,
    metrics: pd.DataFrame,
    evidence: pd.DataFrame,
    loro: pd.DataFrame,
    significance: pd.DataFrame,
    projection: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    def markdown(df: pd.DataFrame, rows: int = 16) -> str:
        if df.empty:
            return "_No rows._"
        view = df.head(rows)
        columns = list(view.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for _, row in view.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    values.append("" if not np.isfinite(value) else f"{value:.4g}")
                else:
                    values.append(str(value).replace("|", "/"))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *body])
    lines = [
        "# ECO-Waste-PCCM Experiment Report",
        "",
        "This report records the final constrained-completion experiments and diagnostics.",
        "",
        "## Algorithmic contribution",
        "- Replaces fixed expert weights with learned convex gating conditioned on variable group and evidence-reliability bin.",
        "- Uses self-supervised validation cells to fit the gate without using held-out test cells.",
        "- Compares fixed, global, group-only, reliability-only, and full group-by-reliability gates.",
        "- Replaces approximate closure normalization with exact Euclidean simplex projection.",
        "- Couples evidence to reconstruction, uncertainty scale, conformal calibration, and acquisition.",
        "",
        "## Projection property audit",
        markdown(projection),
        "",
        "## Evidence utility",
        markdown(evidence),
        "",
        "## Leave-one-risk-component-out",
        markdown(loro),
        "",
        "## Paired significance",
        markdown(significance.sort_values("exact_sign_flip_p")),
        "",
        "## Failure analysis",
        markdown(failures),
        "",
        "## Main interpretation",
        "PCCM is evaluated as a trustworthy constrained completion framework, not a uniformly lowest-error imputer.",
    ]
    (OUT_DIR / "ECO-Waste-PCCM_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    # Repoint reusable helpers to the final experiment directories.
    base.OUT_DIR = OUT_DIR
    base.TABLE_DIR = TABLE_DIR
    base.FIG_DIR = FIG_DIR
    base.PROTOCOL_DIR = PROTOCOL_DIR

    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)

    codebook_aligned, evidence_summary = base.build_codebook_alignment(data)
    codebook_aligned.to_csv(TABLE_DIR / "codebook_aligned.csv", index=False, encoding="utf-8-sig")
    evidence_summary.to_csv(TABLE_DIR / "evidence_quality_summary.csv", index=False, encoding="utf-8-sig")
    data.evidence_calibration_weights.to_csv(TABLE_DIR / "evidence_learning_weights.csv", index=False, encoding="utf-8-sig")
    data.evidence_calibration_diagnostics.to_csv(TABLE_DIR / "evidence_learning_diagnostics.csv", index=False, encoding="utf-8-sig")

    feature_profile = pd.DataFrame(
        {
            "feature": data.features,
            "group": [base.group_id(f) for f in data.features],
            "observed_count": data.observed_mask.sum(axis=0),
            "observed_rate": data.observed_mask.mean(axis=0),
            "mean_evidence_reliability": np.where(
                data.observed_mask.sum(axis=0) > 0,
                (data.reliability * data.observed_mask).sum(axis=0) / np.maximum(data.observed_mask.sum(axis=0), 1),
                np.nan,
            ),
            "is_percent_or_binary": [f in data.percent_features for f in data.features],
            "is_binary": [f in data.binary_features for f in data.features],
        }
    )
    feature_profile.to_csv(TABLE_DIR / "feature_profile.csv", index=False, encoding="utf-8-sig")
    base.build_data_dictionary(data).to_csv(TABLE_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    base.build_missingness_by_group(feature_profile).to_csv(TABLE_DIR / "missingness_by_group.csv", index=False, encoding="utf-8-sig")
    write_split_and_hyperparameter_files(data)

    raw_metrics, metrics, group_raw, final_preds, final_unc = run_imputation_suite_v4(data)
    raw_metrics.to_csv(TABLE_DIR / "imputation_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "imputation_metrics.csv", index=False, encoding="utf-8-sig")
    group_raw.to_csv(TABLE_DIR / "variable_group_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate_metrics(group_raw, ["scenario", "method", "variable_group"]).to_csv(
        TABLE_DIR / "variable_group_metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    raw_metrics[
        [
            "scenario",
            "method",
            "seed",
            "composition_sum_abs_error",
            "treatment_sum_abs_error",
            "range_violation_rate",
            "range_violation_magnitude",
            "mass_balance_relative_error",
        ]
    ].to_csv(TABLE_DIR / "constraint_consistency.csv", index=False, encoding="utf-8-sig")
    significance = paired_significance_tests(raw_metrics)
    significance.to_csv(TABLE_DIR / "paired_significance_tests.csv", index=False, encoding="utf-8-sig")
    failures = failure_analysis(group_raw)
    failures.to_csv(TABLE_DIR / "failure_analysis.csv", index=False, encoding="utf-8-sig")

    ablation = run_ablation_suite_v4(data)
    ablation.to_csv(TABLE_DIR / "model_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    conformal = run_conformal_calibration(data)
    conformal.to_csv(TABLE_DIR / "conformal_calibration.csv", index=False, encoding="utf-8-sig")
    evidence_utility, evidence_bins = run_evidence_utility_analysis(data)
    evidence_utility.to_csv(TABLE_DIR / "evidence_utility_summary.csv", index=False, encoding="utf-8-sig")
    evidence_bins.to_csv(TABLE_DIR / "evidence_reliability_error_bins.csv", index=False, encoding="utf-8-sig")

    main_pred = final_preds.get(ECO_METHOD)
    main_unc = final_unc.get(ECO_METHOD)
    if main_pred is None or main_unc is None:
        train_mask = data.observed_mask & ~base.make_random_mask(data, 0.20, SEEDS[0])
        main_pred, main_unc = pccm_predict(data, train_mask, seed=SEEDS[0])

    completed = raw_prediction(data, main_pred)
    completed.insert(0, "city_code", data.city["city_code"].to_numpy())
    completed.insert(1, "city_name", data.city["city_name"].to_numpy())
    completed.insert(2, "country_name", data.city["country_name"].to_numpy())
    completed.to_csv(TABLE_DIR / "completed_city_matrix_ecowaste_pccm.csv", index=False, encoding="utf-8-sig")

    constraints = base.constraint_audit(data, main_pred)
    constraints.to_csv(TABLE_DIR / "constraint_audit.csv", index=False, encoding="utf-8-sig")
    risk = base.risk_components(data, main_pred, main_unc)
    risk.to_csv(TABLE_DIR / "risk_ranking.csv", index=False, encoding="utf-8-sig")
    recs = base.policy_recommendations(data, risk, main_pred, main_unc)
    recs.to_csv(TABLE_DIR / "policy_recommendations.csv", index=False, encoding="utf-8-sig")
    peer_effects = base.policy_peer_effects(data, risk, main_pred)
    peer_effects.to_csv(TABLE_DIR / "policy_peer_effects.csv", index=False, encoding="utf-8-sig")
    policy_metric_df = policy_metrics(peer_effects, recs, risk)
    policy_metric_df.to_csv(TABLE_DIR / "policy_prioritization_metrics.csv", index=False, encoding="utf-8-sig")
    policy_sensitivity = policy_sensitivity_analysis(peer_effects)
    policy_sensitivity.to_csv(TABLE_DIR / "policy_matching_sensitivity.csv", index=False, encoding="utf-8-sig")
    risk_metrics = run_risk_identification(data, final_preds)
    risk_metrics.to_csv(TABLE_DIR / "risk_identification_metrics.csv", index=False, encoding="utf-8-sig")
    risk_loro = run_leave_one_risk_component_out(data)
    risk_loro.to_csv(TABLE_DIR / "risk_leave_one_component_out.csv", index=False, encoding="utf-8-sig")
    risk_robustness = run_risk_definition_robustness(data, final_preds)
    risk_robustness.to_csv(TABLE_DIR / "risk_definition_robustness.csv", index=False, encoding="utf-8-sig")

    acquisition = active_acquisition_v4(data)
    acquisition.to_csv(TABLE_DIR / "active_acquisition.csv", index=False, encoding="utf-8-sig")
    acquisition_summary_df = active_acquisition_summary(acquisition)
    acquisition_summary_df.to_csv(TABLE_DIR / "active_acquisition_summary.csv", index=False, encoding="utf-8-sig")
    projection = projection_property_audit(data)
    projection.to_csv(TABLE_DIR / "projection_property_audit.csv", index=False, encoding="utf-8-sig")
    gate_df = pd.DataFrame(GATING_RECORDS)
    if not gate_df.empty:
        gate_summary = aggregate_metrics(gate_df, ["gate_mode", "variable_group", "reliability_bin"])
        gate_summary.to_csv(TABLE_DIR / "learned_expert_gating_weights.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(TABLE_DIR / "learned_expert_gating_weights.csv", index=False, encoding="utf-8-sig")

    write_reproducibility_details(data)
    write_experiment_report(data, metrics, evidence_utility, risk_loro, significance, projection, failures)
    summary = {
        "version": "ecowaste_final",
        "method_name": ECO_METHOD,
        "seeds": SEEDS,
        "outputs": str(OUT_DIR),
        "tables": str(TABLE_DIR),
        "figures": str(FIG_DIR),
        "notes": "Implements the rational-analysis recommendations with hierarchical gate ablation, reproducible splits, exact projection, restrained evidence claims, and decision-stability diagnostics.",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
