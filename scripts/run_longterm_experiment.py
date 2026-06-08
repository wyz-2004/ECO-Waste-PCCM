from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
if os.environ.get("ECOWASTE_PROJECT_ROOT"):
    ROOT = Path(os.environ["ECOWASTE_PROJECT_ROOT"]).resolve()
else:
    ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import ecowaste_core as base
import run_experiment as legacy


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
PROTOCOL_DIR = OUT_DIR / "protocol"
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
BASE_METHODS = ["Median", "SoftImpute", "Low-rank SVD", "PCCM"]
PROJECTED_METHODS = [f"{m}+Proj" for m in BASE_METHODS]
SUPPORT_THRESHOLD = 30
EPS = 1e-9


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, PROTOCOL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def scenario_masks(data: base.PreparedData, seed: int) -> dict[str, np.ndarray]:
    scenarios = {
        "random_20pct": base.make_random_mask(data, 0.20, seed),
        "block_by_system": base.make_block_mask(data, seed + 3),
        "region_holdout": base.make_group_holdout_mask(data, "region_id", seed + 5),
        "income_holdout": base.make_group_holdout_mask(data, "income_id_2022", seed + 7),
    }
    rng = np.random.default_rng(seed + 19)
    row_missingness = 1.0 - data.observed_mask.mean(axis=1)
    high_rows = row_missingness >= np.quantile(row_missingness, 0.75)
    high_mask = data.observed_mask & high_rows[:, None] & (rng.random(data.observed_mask.shape) < 0.35)
    scenarios["high_missingness_city"] = high_mask
    return scenarios


def baseline_prediction(
    data: base.PreparedData, train_mask: np.ndarray, method: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if method == "Median":
        pred = legacy.stat_imputer(data, train_mask, "median")
    elif method == "SoftImpute":
        pred = legacy.softimpute(data, train_mask, rank=18, shrink=0.16)
    elif method == "Low-rank SVD":
        pred = base.matrix_factorization(data, train_mask, rank=14, epochs=34, seed=seed, use_evidence=False)
    elif method == "PCCM":
        return legacy.safe_pccm_predict(data, train_mask, seed=seed, use_constraints=False)
    else:
        raise ValueError(method)
    observed = np.where(train_mask, data.model_values, np.nan)
    scale = np.nanstd(observed, axis=0)
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > EPS)])
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback)
    support_gap = 1.0 - train_mask.mean(axis=0)
    uncertainty = np.tile(scale * (0.55 + support_gap), (data.model_values.shape[0], 1))
    return pred, np.maximum(uncertainty, 1e-4)


def projection_metrics(data: base.PreparedData, raw: np.ndarray, projected: np.ndarray) -> dict[str, float]:
    metrics = legacy.constraint_metrics(data, projected)
    metrics["projection_distance"] = float(np.mean(np.abs(projected - raw)))
    metrics["feasibility_score"] = float(
        metrics.get("range_violation_magnitude", 0.0)
        + metrics.get("composition_sum_abs_error", 0.0)
        + metrics.get("treatment_sum_abs_error", 0.0)
    )
    return metrics


def raw_feasibility(data: base.PreparedData, pred: np.ndarray) -> dict[str, float]:
    metrics = legacy.constraint_metrics(data, pred)
    metrics["projection_distance"] = float(np.mean(np.abs(legacy.exact_feasibility_projection(data, pred) - pred)))
    metrics["feasibility_score"] = float(
        metrics.get("range_violation_magnitude", 0.0)
        + metrics.get("composition_sum_abs_error", 0.0)
        + metrics.get("treatment_sum_abs_error", 0.0)
    )
    return metrics


def metric_row(
    data: base.PreparedData,
    pred: np.ndarray,
    test_mask: np.ndarray,
    scenario: str,
    method: str,
    seed: int,
    feasibility: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = base.evaluate(pred, data.model_values, test_mask, None)
    row["percent_mae_points"] = base.percent_mae_pp(data, pred, test_mask)
    row.update(feasibility)
    row.update({"scenario": scenario, "method": method, "seed": seed})
    return row


def run_accuracy_feasibility(
    data: base.PreparedData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    groups = sorted({base.group_id(feature) for feature in data.features})
    for seed in SEEDS:
        for scenario, test_mask in scenario_masks(data, seed).items():
            train_mask = data.observed_mask & ~test_mask
            for i, j in np.argwhere(test_mask):
                mask_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "city_code": data.city.iloc[i]["city_code"],
                        "feature": data.features[j],
                    }
                )
            for method in BASE_METHODS:
                raw, _ = baseline_prediction(data, train_mask, method, seed)
                raw = np.where(np.isfinite(raw), raw, legacy.stat_imputer(data, train_mask, "median"))
                projected = legacy.exact_feasibility_projection(data, raw)
                rows.append(metric_row(data, raw, test_mask, scenario, method, seed, raw_feasibility(data, raw)))
                rows.append(
                    metric_row(
                        data,
                        projected,
                        test_mask,
                        scenario,
                        f"{method}+Proj",
                        seed,
                        projection_metrics(data, raw, projected),
                    )
                )
                if scenario == "random_20pct":
                    for group in groups:
                        group_mask = test_mask & np.array(
                            [base.group_id(feature) == group for feature in data.features]
                        )[None, :]
                        result = base.evaluate(projected, data.model_values, group_mask, None)
                        result.update(
                            {
                                "seed": seed,
                                "method": f"{method}+Proj",
                                "variable_group": group,
                                "evaluated_cells": int(group_mask.sum()),
                            }
                        )
                        group_rows.append(result)
    return pd.DataFrame(rows), pd.DataFrame(group_rows), pd.DataFrame(mask_rows)


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [
        col
        for col in df.columns
        if col not in set(keys + ["seed"]) and pd.api.types.is_numeric_dtype(df[col])
    ]
    grouped = df.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().reset_index()
    std = grouped[numeric].std(ddof=1).add_suffix("_std").reset_index()
    std = std.rename(columns={f"{key}_std": key for key in keys})
    count = grouped.size().rename("seeds").reset_index()
    return mean.merge(std, on=keys, how="left").merge(count, on=keys, how="left")


def paired_bootstrap(raw: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260606)
    rows: list[dict[str, object]] = []
    for scenario in sorted(raw["scenario"].unique()):
        for baseline in ["Median+Proj", "SoftImpute+Proj", "Low-rank SVD+Proj"]:
            for metric in ["mae_norm", "rmse_norm"]:
                eco = raw[(raw.scenario == scenario) & (raw.method == "PCCM+Proj")].set_index("seed")[metric]
                other = raw[(raw.scenario == scenario) & (raw.method == baseline)].set_index("seed")[metric]
                paired = pd.concat([eco.rename("eco"), other.rename("baseline")], axis=1).dropna()
                diff = paired["eco"].to_numpy() - paired["baseline"].to_numpy()
                boot = np.array(
                    [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)]
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "comparison": f"PCCM+Proj vs {baseline}",
                        "metric": metric,
                        "paired_mean_difference": float(diff.mean()),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "standardized_effect": float(diff.mean() / max(diff.std(ddof=1), EPS)),
                        "seeds": len(diff),
                    }
                )
    return pd.DataFrame(rows)


def calibration_metrics(
    data: base.PreparedData,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    calib_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float]:
    return legacy.conformal_from_masks(
        data, pred, np.maximum(uncertainty, 1e-4), calib_mask, test_mask, reliability_weighted=False
    )


def safe_evidence_calibration_metrics(
    data: base.PreparedData,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    calib_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    use_evidence: bool,
) -> dict[str, float]:
    if not use_evidence:
        return calibration_metrics(data, pred, uncertainty, calib_mask, test_mask)

    def evidence_conformal(fit: np.ndarray, evaluate: np.ndarray, bins: int, shrink: float, beta: float) -> dict[str, float]:
        gap = 1.0 - np.clip(np.nan_to_num(data.reliability, nan=0.55), 0.08, 1.0)
        centered = gap - float(np.mean(gap[fit]))
        scaled = np.maximum(uncertainty * np.exp(beta * centered), 1e-4)
        fit_error = np.abs(pred[fit] - data.model_values[fit])
        eval_error = np.abs(pred[evaluate] - data.model_values[evaluate])
        fit_score = fit_error / scaled[fit]
        fit_gap = gap[fit]
        eval_gap = gap[evaluate]
        thresholds = np.quantile(fit_gap, np.linspace(0, 1, bins + 1)[1:-1]) if bins > 1 else np.array([])
        fit_bin = np.digitize(fit_gap, thresholds)
        eval_bin = np.digitize(eval_gap, thresholds)
        out: dict[str, float] = {}
        ece_terms = []
        for target in [0.50, 0.80, 0.90]:
            global_q = float(np.quantile(fit_score, target))
            widths = np.zeros(eval_error.shape, dtype=float)
            for bin_id in range(bins):
                local = fit_score[fit_bin == bin_id]
                local_q = float(np.quantile(local, target)) if len(local) >= 35 else global_q
                q = shrink * local_q + (1.0 - shrink) * global_q
                widths[eval_bin == bin_id] = 2.0 * q * scaled[evaluate][eval_bin == bin_id]
            coverage = float(np.mean(eval_error <= widths / 2.0))
            out[f"coverage_{int(target * 100)}"] = coverage
            out[f"width_{int(target * 100)}"] = float(np.mean(widths))
            ece_terms.append(abs(coverage - target))
        out["ece"] = float(np.mean(ece_terms))
        out["crps_proxy"] = float(np.mean(np.minimum(eval_error, np.quantile(eval_error, 0.95))))
        return out

    rng = np.random.default_rng(seed + 881)
    tune_mask = calib_mask & (rng.random(calib_mask.shape) < 0.45)
    fit_mask = calib_mask & ~tune_mask
    if tune_mask.sum() < 100 or fit_mask.sum() < 100:
        return calibration_metrics(data, pred, uncertainty, calib_mask, test_mask)
    best_config = (1, 0.0, 0.0)
    baseline = evidence_conformal(fit_mask, tune_mask, *best_config)
    best_loss = float(baseline["ece"] + 0.002 * baseline["width_80"])
    for bins in [2, 3]:
        for shrink in [0.35, 0.60, 0.85]:
            for beta in [-0.50, -0.25, 0.0, 0.25, 0.50, 0.75]:
                candidate = evidence_conformal(fit_mask, tune_mask, bins, shrink, beta)
                loss = float(candidate["ece"] + 0.002 * candidate["width_80"])
                if loss < best_loss * 0.985:
                    best_loss = loss
                    best_config = (bins, shrink, beta)
    result = evidence_conformal(calib_mask, test_mask, *best_config)
    result["evidence_bins"] = best_config[0]
    result["evidence_shrinkage"] = best_config[1]
    result["evidence_scale_beta"] = best_config[2]
    return result


def run_calibration_fairness(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 400)
        test_mask = base.make_random_mask(data, 0.20, seed + 401)
        pool = data.observed_mask & ~test_mask
        calib_mask = pool & (rng.random(pool.shape) < 0.15)
        train_mask = pool & ~calib_mask
        for method in BASE_METHODS:
            raw, unc = baseline_prediction(data, train_mask, method, seed + 17)
            projected = legacy.exact_feasibility_projection(data, raw)
            for name, pred in [(method, raw), (f"{method}+Proj", projected)]:
                result = calibration_metrics(data, pred, unc, calib_mask, test_mask)
                result.update({"seed": seed, "method": name})
                rows.append(result)
    return pd.DataFrame(rows)


def evidence_variant(
    data: base.PreparedData, train_mask: np.ndarray, seed: int, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    original = data.reliability.copy()
    try:
        if variant == "No evidence":
            data.reliability = np.full_like(original, 0.55)
            return legacy.safe_pccm_predict(data, train_mask, seed=seed, use_evidence=False)
        if variant == "Shuffled evidence":
            rng = np.random.default_rng(seed + 91)
            data.reliability = rng.permutation(original.ravel()).reshape(original.shape)
        elif variant == "Heuristic evidence":
            data.reliability = data.static_reliability.copy()
        elif variant != "Learned evidence":
            raise ValueError(variant)
        return legacy.safe_pccm_predict(data, train_mask, seed=seed, use_evidence=True)
    finally:
        data.reliability = original


def run_evidence_ablation(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = ["No evidence", "Shuffled evidence", "Heuristic evidence", "Learned evidence"]
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 700)
        test_mask = base.make_random_mask(data, 0.20, seed + 701)
        pool = data.observed_mask & ~test_mask
        calib_mask = pool & (rng.random(pool.shape) < 0.15)
        train_mask = pool & ~calib_mask
        for variant in variants:
            pred, unc = evidence_variant(data, train_mask, seed + 37, variant)
            result = base.evaluate(pred, data.model_values, test_mask, unc)
            result.update(
                safe_evidence_calibration_metrics(
                    data,
                    pred,
                    unc,
                    calib_mask,
                    test_mask,
                    seed,
                    use_evidence=variant != "No evidence",
                )
            )
            result.update({"seed": seed, "variant": variant})
            rows.append(result)
    return pd.DataFrame(rows)


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    return np.nan_to_num((matrix - low) / np.maximum(high - low, EPS), nan=0.0)


def normalize_from_reference(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanmin(reference, axis=0)
    high = np.nanmax(reference, axis=0)
    scale = np.maximum(high - low, EPS)
    return (
        np.nan_to_num((reference - low) / scale, nan=0.0),
        np.nan_to_num((values - low) / scale, nan=0.0),
    )


def fit_nonnegative_score(features: np.ndarray, target: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    x = normalize_columns(features)
    y = (target - target.min()) / max(target.max() - target.min(), EPS)
    weights = np.full(x.shape[1], 1.0 / x.shape[1])
    lipschitz = 2.0 * np.linalg.norm(x, ord=2) ** 2 / max(len(y), 1) + 2.0 * ridge
    step = 1.0 / max(lipschitz, EPS)
    for _ in range(1200):
        grad = 2.0 * x.T @ (x @ weights - y) / max(len(y), 1) + 2.0 * ridge * weights
        updated = np.maximum(0.0, weights - step * grad)
        if np.linalg.norm(updated - weights) < 1e-9:
            break
        weights = updated
    total = weights.sum()
    return weights / total if total > EPS else np.full_like(weights, 1.0 / len(weights))


def acquisition_feature_matrix(
    entries: np.ndarray,
    uncertainty: np.ndarray,
    evidence_gap: np.ndarray,
    variable_missingness: np.ndarray,
    constraint_impact: np.ndarray,
    city_risk: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            [
                uncertainty[i, j],
                evidence_gap[i, j],
                variable_missingness[j],
                constraint_impact[j],
                city_risk[i],
                uncertainty[i, j] * evidence_gap[i, j],
            ]
            for i, j in entries
        ],
        dtype=float,
    )


def run_active_acquisition(data: base.PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    risk_cols = [data.features.index(feature) for feature in risk_vars if feature in data.features]
    col_mask = np.zeros(len(data.features), dtype=bool)
    col_mask[risk_cols] = True
    budgets = [5, 10, 20, 40, 80, 120]
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 1100)
        candidates = data.observed_mask & col_mask[None, :] & (rng.random(data.observed_mask.shape) < 0.42)
        calibration = (
            data.observed_mask
            & col_mask[None, :]
            & ~candidates
            & (rng.random(data.observed_mask.shape) < 0.18)
        )
        train_mask = data.observed_mask & ~candidates & ~calibration
        pred, unc = legacy.safe_pccm_predict(data, train_mask, seed=seed + 111)
        candidate_entries = np.argwhere(candidates)
        calibration_entries = np.argwhere(calibration)
        variable_missingness = 1.0 - train_mask.mean(axis=0)
        evidence_gap = 1.0 - np.clip(np.nan_to_num(data.reliability, nan=0.35), 0.0, 1.0)
        constraint_impact = np.array(
            [1.0 if base.group_id(feature) in {"composition", "treatment", "collection"} else 0.25 for feature in data.features]
        )
        risk_frame = base.risk_components(data, pred, unc).set_index("city_code")
        city_risk = risk_frame.loc[data.city["city_code"], "risk_percentile"].to_numpy(dtype=float)
        x_cal = acquisition_feature_matrix(
            calibration_entries, unc, evidence_gap, variable_missingness, constraint_impact, city_risk
        )
        y_cal = np.abs(pred[calibration] - data.model_values[calibration])
        no_evidence_cols = [0, 2, 3, 4]
        evidence_cols = [1, 5]
        x_candidate_raw = acquisition_feature_matrix(
            candidate_entries, unc, evidence_gap, variable_missingness, constraint_impact, city_risk
        )
        x_cal_norm, x_candidate = normalize_from_reference(x_cal, x_candidate_raw)
        full_weights = fit_nonnegative_score(x_cal_norm, y_cal)
        no_ev_weights = fit_nonnegative_score(x_cal_norm[:, no_evidence_cols], y_cal)
        no_ev_cal = x_cal_norm[:, no_evidence_cols] @ no_ev_weights
        y_norm = (y_cal - y_cal.min()) / max(y_cal.max() - y_cal.min(), EPS)
        incremental_target = np.maximum(0.0, y_norm - no_ev_cal)
        evidence_residual_weights = fit_nonnegative_score(x_cal_norm[:, evidence_cols], incremental_target)
        for feature, value in zip(
            ["uncertainty", "evidence_gap", "missingness", "constraint_impact", "city_risk", "uncertainty_x_evidence"],
            full_weights,
        ):
            weight_rows.append({"seed": seed, "model": "Learned ECO-Acquire", "feature": feature, "weight": value})
        candidate_error = np.abs(pred[candidates] - data.model_values[candidates])
        reference_pred = pred.copy()
        reference_pred[candidates] = data.model_values[candidates]
        reference_rank = rank(legacy.risk_score_array(data, reference_pred))
        reference_top20 = set(np.argsort(reference_rank)[-20:])
        strategies = {
            "Random": rng.random(len(candidate_entries)),
            "Uncertainty-only": x_candidate[:, 0],
            "Evidence-gap-only": x_candidate[:, 1],
            "Constraint-only": x_candidate[:, 3],
            "Learned ECO-Acquire": x_candidate @ full_weights,
            "Learned no-evidence": x_candidate[:, no_evidence_cols] @ no_ev_weights,
        }
        no_ev_candidate = strategies["Learned no-evidence"]
        full_cal = x_cal_norm @ full_weights
        full_candidate = x_candidate @ full_weights
        for budget in budgets:
            calibration_budget = max(1, int(round(budget / max(len(candidate_entries), 1) * len(calibration_entries))))
            baseline_order = np.argsort(no_ev_cal)[::-1][:calibration_budget]
            evidence_order = np.argsort(full_cal)[::-1][:calibration_budget]
            no_ev_capture = float(y_cal[baseline_order].sum())
            evidence_capture = float(y_cal[evidence_order].sum())
            # Evidence is too noisy for scarce-cell decisions. The safe policy
            # nests the no-evidence score below 120 cells and only enables the
            # evidence residual at the largest audited budget after validation.
            use_evidence_score = budget >= 120 and evidence_capture > no_ev_capture * 1.01
            strategies["Safe evidence ECO-Acquire"] = full_candidate if use_evidence_score else no_ev_candidate
            weight_rows.append(
                {
                    "seed": seed,
                    "model": "Safe evidence ECO-Acquire",
                    "feature": f"use_evidence_budget_{budget}",
                    "weight": float(use_evidence_score),
                }
            )
            for strategy, scores in strategies.items():
                order = np.argsort(scores)[::-1]
                selected_idx = order[: min(budget, len(order))]
                selected = candidate_entries[selected_idx]
                keep = np.ones(len(candidate_entries), dtype=bool)
                keep[selected_idx] = False
                updated = pred.copy()
                if len(selected):
                    updated[selected[:, 0], selected[:, 1]] = data.model_values[selected[:, 0], selected[:, 1]]
                updated_rank = rank(legacy.risk_score_array(data, updated))
                updated_top20 = set(np.argsort(updated_rank)[-20:])
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "budget_cells": budget,
                        "remaining_mae": float(candidate_error[keep].mean()) if keep.any() else 0.0,
                        "error_capture_rate": float(candidate_error[selected_idx].sum() / max(candidate_error.sum(), EPS)),
                        "risk_rank_correlation": float(np.corrcoef(reference_rank, updated_rank)[0, 1]),
                        "top20_overlap": len(reference_top20 & updated_top20) / 20.0,
                        "candidate_cells": len(candidate_entries),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def build_table4(group_raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_df in group_raw.groupby("variable_group"):
        method_stats = (
            group_df.groupby("method")
            .agg(
                mae_norm=("mae_norm", "mean"),
                mae_std=("mae_norm", "std"),
                evaluated_cells=("evaluated_cells", "sum"),
                seeds_with_support=("evaluated_cells", lambda values: int((values > 0).sum())),
            )
            .reset_index()
        )
        support = int(method_stats["evaluated_cells"].max())
        supported_seeds = int(method_stats["seeds_with_support"].max())
        eligible = support >= SUPPORT_THRESHOLD and supported_seeds >= 8
        winner = method_stats.sort_values("mae_norm").iloc[0] if eligible else None
        pccm = method_stats[method_stats.method == "PCCM+Proj"].iloc[0]
        rows.append(
            {
                "variable_group": group,
                "support_cells": support,
                "seeds_with_support": supported_seeds,
                "support_status": "Supported" if eligible else "Insufficient support",
                "winner": winner["method"] if winner is not None else "Not assigned",
                "best_mae": float(winner["mae_norm"]) if winner is not None else np.nan,
                "pccm_mae": float(pccm["mae_norm"]),
                "pccm_mae_std": float(pccm["mae_std"]),
                "pccm_delta_from_best": float(pccm["mae_norm"] - winner["mae_norm"]) if winner is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["support_status", "support_cells"], ascending=[False, False])


def write_report(
    accuracy: pd.DataFrame,
    calibration: pd.DataFrame,
    evidence: pd.DataFrame,
    acquisition: pd.DataFrame,
    table4: pd.DataFrame,
) -> None:
    def compact_table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}")

    random_projected = accuracy[
        (accuracy.scenario == "random_20pct") & accuracy.method.isin(PROJECTED_METHODS)
    ].sort_values("rmse_norm")
    learned = evidence[evidence.variant == "Learned evidence"].iloc[0]
    shuffled = evidence[evidence.variant == "Shuffled evidence"].iloc[0]
    eco = acquisition[acquisition.strategy == "Learned ECO-Acquire"]
    lines = [
        "# ECO-Waste final AAAI experiment",
        "",
        "This focused experiment implements the final protocol: 10 shared seeds, projected baselines, fair conformal calibration, evidence ablations, multi-budget acquisition, and support-aware variable-group reporting.",
        "",
        "## Accuracy-feasibility trade-off",
        "```text",
        compact_table(
            random_projected,
            ["method", "mae_norm", "rmse_norm", "projection_distance", "feasibility_score"],
        ),
        "```",
        "",
        "## Evidence/reliability",
        f"Learned evidence ECE={learned.ece:.4f}; shuffled evidence ECE={shuffled.ece:.4f}. Evidence is retained only to the extent supported by calibration and acquisition results.",
        "",
        "## Active acquisition",
        "```text",
        compact_table(
            eco,
            ["budget_cells", "remaining_mae", "error_capture_rate", "risk_rank_correlation", "top20_overlap"],
        ),
        "```",
        "",
        "## Table 4 support rule",
        f"Variable groups require at least {SUPPORT_THRESHOLD} total held-out cells and support in at least 8/10 seeds before a winner is assigned.",
        "```text",
        table4[
            ["variable_group", "support_cells", "seeds_with_support", "support_status", "winner", "best_mae", "pccm_mae"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
    ]
    (OUT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    base.OUT_DIR = OUT_DIR
    base.TABLE_DIR = TABLE_DIR
    base.PROTOCOL_DIR = PROTOCOL_DIR
    legacy.GATING_RECORDS.clear()
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)

    if args.active_only:
        acquisition_raw, acquisition_weights = run_active_acquisition(data)
        acquisition = aggregate(acquisition_raw, ["strategy", "budget_cells"])
        acquisition_raw.to_csv(TABLE_DIR / "active_acquisition_by_seed.csv", index=False, encoding="utf-8-sig")
        acquisition.to_csv(TABLE_DIR / "active_acquisition_summary.csv", index=False, encoding="utf-8-sig")
        aggregate(acquisition_weights, ["model", "feature"]).to_csv(
            TABLE_DIR / "active_acquisition_weights.csv", index=False, encoding="utf-8-sig"
        )
        print(acquisition[acquisition.strategy == "Safe evidence ECO-Acquire"].to_string(index=False))
        return

    raw, group_raw, masks = run_accuracy_feasibility(data)
    summary = aggregate(raw, ["scenario", "method"])
    raw.to_csv(TABLE_DIR / "accuracy_feasibility_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "accuracy_feasibility_summary.csv", index=False, encoding="utf-8-sig")
    masks.to_csv(PROTOCOL_DIR / "shared_mask_assignments.csv", index=False, encoding="utf-8-sig")
    paired_bootstrap(raw).to_csv(TABLE_DIR / "paired_bootstrap_effects.csv", index=False, encoding="utf-8-sig")

    group_raw.to_csv(TABLE_DIR / "variable_group_by_seed.csv", index=False, encoding="utf-8-sig")
    table4 = build_table4(group_raw)
    table4.to_csv(TABLE_DIR / "table4_variable_group_support.csv", index=False, encoding="utf-8-sig")

    calibration_raw = run_calibration_fairness(data)
    calibration = aggregate(calibration_raw, ["method"])
    calibration_raw.to_csv(TABLE_DIR / "calibration_fairness_by_seed.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(TABLE_DIR / "calibration_fairness_summary.csv", index=False, encoding="utf-8-sig")

    evidence_raw = run_evidence_ablation(data)
    evidence = aggregate(evidence_raw, ["variant"])
    evidence_raw.to_csv(TABLE_DIR / "evidence_ablation_by_seed.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(TABLE_DIR / "evidence_ablation_summary.csv", index=False, encoding="utf-8-sig")

    acquisition_raw, acquisition_weights = run_active_acquisition(data)
    acquisition = aggregate(acquisition_raw, ["strategy", "budget_cells"])
    acquisition_raw.to_csv(TABLE_DIR / "active_acquisition_by_seed.csv", index=False, encoding="utf-8-sig")
    acquisition.to_csv(TABLE_DIR / "active_acquisition_summary.csv", index=False, encoding="utf-8-sig")
    aggregate(acquisition_weights, ["model", "feature"]).to_csv(
        TABLE_DIR / "active_acquisition_weights.csv", index=False, encoding="utf-8-sig"
    )

    gate = pd.DataFrame(legacy.GATING_RECORDS)
    if not gate.empty:
        aggregate(gate, ["gate_mode", "variable_group", "reliability_bin"]).to_csv(
            TABLE_DIR / "learned_gate_weights.csv", index=False, encoding="utf-8-sig"
        )

    write_report(summary, calibration, evidence, acquisition, table4)
    metadata = {
        "version": "ecowaste_longterm",
        "seeds": SEEDS,
        "support_threshold": SUPPORT_THRESHOLD,
        "projected_baselines": PROJECTED_METHODS,
        "outputs": str(OUT_DIR),
        "claim_boundary": "verification and acquisition priority; not causal policy recommendation",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
