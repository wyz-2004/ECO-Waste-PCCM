from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = Path(os.environ["ECOWASTE_PROJECT_ROOT"]).resolve() if os.environ.get("ECOWASTE_PROJECT_ROOT") else SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import ecowaste_core as base
import run_experiment as legacy


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
ABLATION_SEEDS = SEEDS[:5]
EPS = 1e-9
STRONG_METHODS = [
    "Median",
    "KNN",
    "MICE",
    "MissForest",
    "SoftImpute",
    "Low-rank SVD",
    "GAIN-style",
    "Ours-L1",
    "Ours-L2",
]


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [c for c in frame.columns if c not in set(keys + ["seed"]) and pd.api.types.is_numeric_dtype(frame[c])]
    grouped = frame.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().reset_index()
    std = grouped[numeric].std(ddof=1).add_suffix("_std").reset_index()
    std = std.rename(columns={f"{key}_std": key for key in keys})
    return mean.merge(std, on=keys, how="left").merge(grouped.size().rename("seeds").reset_index(), on=keys, how="left")


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank correlation without an optional SciPy dependency."""
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if np.std(x_rank) <= EPS or np.std(y_rank) <= EPS:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def generic_uncertainty(data: base.PreparedData, train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    observed = np.where(train, data.model_values, np.nan)
    scale = np.nanstd(observed, axis=0)
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > EPS)])
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback if np.isfinite(fallback) else 1.0)
    feature_gap = 1.0 - train.mean(axis=0)
    row_gap = 1.0 - train.mean(axis=1)
    disagreement = np.abs(pred - legacy.stat_imputer(data, train, "median"))
    return np.maximum(0.45 * disagreement + scale[None, :] * (0.35 + feature_gap[None, :] + 0.20 * row_gap[:, None]), 1e-4)


def predict(data: base.PreparedData, train: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "Median":
        pred = legacy.stat_imputer(data, train, "median")
        unc = generic_uncertainty(data, train, pred)
    elif method == "KNN":
        pred = base.weighted_city_knn(data, train, k=10)
        unc = generic_uncertainty(data, train, pred)
    elif method == "MICE":
        pred = legacy.mice_ridge_imputer(data, train, iterations=4, max_features=42)
        unc = generic_uncertainty(data, train, pred)
    elif method == "MissForest":
        pred = legacy.missforest_imputer(data, train, seed=seed, iterations=2, trees=5)
        unc = generic_uncertainty(data, train, pred)
    elif method == "SoftImpute":
        pred = legacy.softimpute(data, train, rank=18, shrink=0.16, epochs=32)
        unc = generic_uncertainty(data, train, pred)
    elif method == "Low-rank SVD":
        pred = base.matrix_factorization(data, train, rank=14, epochs=34, seed=seed, use_evidence=False)
        unc = generic_uncertainty(data, train, pred)
    elif method == "GAIN-style":
        pred = legacy.gain_style_imputer(data, train, seed=seed)
        unc = generic_uncertainty(data, train, pred)
    elif method == "Ours-L1":
        pred, unc = legacy.safe_pccm_predict(data, train, seed=seed, use_evidence=False, use_constraints=False)
    elif method == "Ours-L2":
        pred, unc = legacy.pccm_l2_predict(data, train, seed=seed, use_constraints=False)
    else:
        raise ValueError(method)
    fallback = legacy.stat_imputer(data, train, "median")
    return np.where(np.isfinite(pred), pred, fallback), np.maximum(np.nan_to_num(unc, nan=1.0), 1e-4)


def violation(data: base.PreparedData, pred: np.ndarray) -> float:
    metrics = legacy.constraint_metrics(data, pred)
    return float(
        metrics.get("range_violation_magnitude", 0.0)
        + metrics.get("composition_sum_abs_error", 0.0)
        + metrics.get("treatment_sum_abs_error", 0.0)
    )


def benchmark_row(data: base.PreparedData, pred: np.ndarray, test: np.ndarray, method: str, seed: int) -> dict[str, object]:
    metrics: dict[str, object] = base.evaluate(pred, data.model_values, test, None)
    metrics["violation"] = violation(data, pred)
    metrics["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
    metrics.update({"method": method, "seed": seed})
    return metrics


def city_strong_benchmark(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        test = base.make_random_mask(data, 0.20, seed)
        train = data.observed_mask & ~test
        for method in STRONG_METHODS:
            raw, _ = predict(data, train, method, seed + 17)
            projected = legacy.exact_feasibility_projection(data, raw)
            rows.append(benchmark_row(data, projected, test, method, seed))
    return pd.DataFrame(rows)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260608)
    rows = []
    for ours, metric in [("Ours-L1", "mae_norm"), ("Ours-L2", "rmse_norm")]:
        left = frame[frame.method == ours].set_index("seed")[metric]
        for baseline in [m for m in STRONG_METHODS if not m.startswith("Ours")]:
            right = frame[frame.method == baseline].set_index("seed")[metric]
            paired = pd.concat([left.rename("ours"), right.rename("baseline")], axis=1).dropna()
            diff = paired.ours.to_numpy() - paired.baseline.to_numpy()
            boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(3000)])
            rows.append(
                {
                    "ours": ours,
                    "baseline": baseline,
                    "metric": metric,
                    "paired_mean_difference": float(diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "wins": int((diff < -1e-10).sum()),
                    "ties": int((np.abs(diff) <= 1e-10).sum()),
                    "losses": int((diff > 1e-10).sum()),
                }
            )
    return pd.DataFrame(rows)


def common_acquisition_mae(data: base.PreparedData, pred: np.ndarray, unc: np.ndarray, test: np.ndarray, budget: int = 120) -> float:
    entries = np.argwhere(test)
    errors = np.abs(pred[test] - data.model_values[test])
    if len(entries) <= budget:
        return 0.0
    scores = unc[test]
    selected = np.argsort(scores)[::-1][:budget]
    keep = np.ones(len(entries), dtype=bool)
    keep[selected] = False
    return float(errors[keep].mean())


def component_ablation(data: base.PreparedData) -> pd.DataFrame:
    variants: list[tuple[str, dict[str, object]]] = [
        ("Full Ours-L1", {}),
        ("Ours-L2 only", {"head": "l2"}),
        ("Uniform expert weights", {"use_adaptive_stack": False}),
        ("w/o cross-fitting", {"use_cross_fitting": False}),
        ("w/o graph expert", {"use_graph": False}),
        ("w/o low-rank expert", {"use_low_rank": False}),
        ("w/o SoftImpute expert", {"use_softimpute": False}),
        ("Unweighted projection", {"projection_mode": "unweighted"}),
        ("w/o projection", {"use_constraints": False}),
        ("Forced evidence (no gate)", {"forced_evidence": True}),
    ]
    rows: list[dict[str, object]] = []
    for seed in ABLATION_SEEDS:
        test = base.make_random_mask(data, 0.20, seed + 2200)
        rng = np.random.default_rng(seed + 2201)
        pool = data.observed_mask & ~test
        calib = pool & (rng.random(pool.shape) < 0.15)
        train = pool & ~calib
        for variant, config in variants:
            options = dict(config)
            head = options.pop("head", "l1")
            forced = bool(options.pop("forced_evidence", False))
            if head == "l2":
                raw, unc = legacy.pccm_l2_predict(data, train, seed=seed + 31, use_constraints=True)
            else:
                raw, unc = legacy.safe_pccm_predict(
                    data,
                    train,
                    seed=seed + 31,
                    use_evidence=False,
                    **options,
                )
            if forced:
                gap = 1.0 - np.clip(np.nan_to_num(data.reliability, nan=0.55), 0.08, 1.0)
                centered = gap - float(np.mean(gap[calib]))
                unc = np.maximum(unc * np.exp(0.75 * centered), 1e-4)
            result: dict[str, object] = base.evaluate(raw, data.model_values, test, None)
            result.update(legacy.conformal_from_masks(data, raw, unc, calib, test, reliability_weighted=False))
            result["violation"] = violation(data, raw)
            result["acquisition_mae"] = common_acquisition_mae(data, raw, unc, test)
            result.update({"variant": variant, "seed": seed})
            rows.append(result)
    return pd.DataFrame(rows)


def evidence_semantic_test(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    learned = data.reliability.copy()
    heuristic = data.static_reliability.copy()
    for seed in SEEDS:
        test = base.make_random_mask(data, 0.20, seed + 3300)
        train = data.observed_mask & ~test
        pred, _ = legacy.safe_pccm_predict(data, train, seed=seed + 41, use_evidence=False)
        errors = np.abs(pred[test] - data.model_values[test])
        rng = np.random.default_rng(seed + 3301)
        variants = {
            "Learned evidence": learned,
            "Heuristic evidence": heuristic,
            "Shuffled evidence": rng.permutation(learned.ravel()).reshape(learned.shape),
        }
        for variant, reliability in variants.items():
            gap = 1.0 - np.clip(np.nan_to_num(reliability[test], nan=0.55), 0.0, 1.0)
            corr = spearman_correlation(gap, errors)
            low, high = np.quantile(gap, [0.25, 0.75])
            low_error = float(errors[gap <= low].mean())
            high_error = float(errors[gap >= high].mean())
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "spearman_error_correlation": corr,
                    "high_to_low_error_ratio": high_error / max(low_error, EPS),
                    "high_gap_error": high_error,
                    "low_gap_error": low_error,
                }
            )
    return pd.DataFrame(rows)


def safety_ablation() -> pd.DataFrame:
    raw = pd.read_csv(TABLE_DIR / "active_acquisition_by_seed.csv", encoding="utf-8-sig")
    keep = raw[raw.strategy.isin(["Safe evidence ECO-Acquire", "Learned ECO-Acquire", "Learned no-evidence", "Random"])].copy()
    baseline = keep[keep.strategy == "Learned no-evidence"][["seed", "budget_cells", "remaining_mae"]].rename(
        columns={"remaining_mae": "no_evidence_mae"}
    )
    keep = keep.merge(baseline, on=["seed", "budget_cells"], how="left")
    keep["regret_vs_no_evidence"] = keep.remaining_mae - keep.no_evidence_mae
    by_seed = (
        keep.groupby(["strategy", "seed"], as_index=False)
        .agg(
            mean_remaining_mae=("remaining_mae", "mean"),
            worst_regret_vs_no_evidence=("regret_vs_no_evidence", "max"),
            mean_rank_correlation=("risk_rank_correlation", "mean"),
        )
    )
    return aggregate(by_seed, ["strategy"])


def country_protocol_summary() -> pd.DataFrame:
    summary = pd.read_csv(TABLE_DIR / "country_generalization_summary.csv", encoding="utf-8-sig")
    raw = pd.read_csv(TABLE_DIR / "country_generalization_by_seed.csv", encoding="utf-8-sig")
    projected = summary[summary.method.str.endswith("+Proj")].copy()
    raw_projected = raw[raw.method.str.endswith("+Proj")].copy()
    rows = []
    for scenario, subset in projected.groupby("scenario"):
        ours = subset[subset.method == "PCCM+Proj"].iloc[0]
        baselines = subset[subset.method != "PCCM+Proj"]
        best_rmse = baselines.sort_values("rmse_norm").iloc[0]
        best_mae = baselines.sort_values("mae_norm").iloc[0]
        ours_seed = raw_projected[(raw_projected.scenario == scenario) & (raw_projected.method == "PCCM+Proj")].set_index("seed")
        baseline_seed = raw_projected[
            (raw_projected.scenario == scenario) & (raw_projected.method == best_rmse.method)
        ].set_index("seed")
        diff = ours_seed.rmse_norm - baseline_seed.rmse_norm
        rows.append(
            {
                "scenario": scenario,
                "best_baseline_rmse_method": best_rmse.method,
                "best_baseline_rmse": best_rmse.rmse_norm,
                "ours_rmse": ours.rmse_norm,
                "delta_rmse": ours.rmse_norm - best_rmse.rmse_norm,
                "best_baseline_mae_method": best_mae.method,
                "best_baseline_mae": best_mae.mae_norm,
                "ours_mae": ours.mae_norm,
                "delta_mae": ours.mae_norm - best_mae.mae_norm,
                "rmse_wins": int((diff < -1e-10).sum()),
                "rmse_ties": int((np.abs(diff) <= 1e-10).sum()),
                "rmse_losses": int((diff > 1e-10).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "strong", "ablation", "semantics", "derived"],
        default="all",
    )
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)

    if args.stage in {"all", "strong"}:
        strong = city_strong_benchmark(data)
        strong.to_csv(TABLE_DIR / "city_strong_benchmark_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(strong, ["method"]).to_csv(TABLE_DIR / "city_strong_benchmark_summary.csv", index=False, encoding="utf-8-sig")
        paired_effects(strong).to_csv(TABLE_DIR / "city_strong_paired_effects.csv", index=False, encoding="utf-8-sig")

    if args.stage in {"all", "ablation"}:
        ablation = component_ablation(data)
        ablation.to_csv(TABLE_DIR / "component_ablation_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(ablation, ["variant"]).to_csv(TABLE_DIR / "component_ablation_summary.csv", index=False, encoding="utf-8-sig")

    if args.stage in {"all", "semantics"}:
        semantics = evidence_semantic_test(data)
        semantics.to_csv(TABLE_DIR / "evidence_semantic_test_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(semantics, ["variant"]).to_csv(TABLE_DIR / "evidence_semantic_test_summary.csv", index=False, encoding="utf-8-sig")

    if args.stage in {"all", "derived"}:
        safety_ablation().to_csv(TABLE_DIR / "safety_ablation_summary.csv", index=False, encoding="utf-8-sig")
        country_protocol_summary().to_csv(TABLE_DIR / "country_protocol_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "city_strong_methods": STRONG_METHODS,
        "ablation_seeds": ABLATION_SEEDS,
        "independent_external_data": "Not claimed; country and city workbooks are from the same data program.",
    }
    (OUT_DIR / "experiment_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.stage in {"all", "strong"}:
        print(aggregate(strong, ["method"])[["method", "mae_norm", "rmse_norm", "violation"]].sort_values("rmse_norm").to_string(index=False))


if __name__ == "__main__":
    main()
