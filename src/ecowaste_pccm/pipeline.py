from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constraints import closure_group_for_feature, closure_violation, project_workbook
from .data import WorkbookData, feature_bounds, feature_kind, group_keys, load_city_workbook
from .metrics import masked_errors
from .model import denormalize, fit_pccm


@dataclass
class PipelineResult:
    output_dir: Path
    metrics: dict[str, float]


def _sample_mask(observed: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    entries = np.argwhere(observed)
    n = max(1, int(round(len(entries) * fraction)))
    chosen = rng.choice(len(entries), size=n, replace=False)
    mask = np.zeros_like(observed, dtype=bool)
    selected = entries[chosen]
    mask[selected[:, 0], selected[:, 1]] = True
    return mask


def _log_transform_features(features: list[str], values: np.ndarray) -> np.ndarray:
    flags = []
    for j, feature in enumerate(features):
        kind = feature_kind(feature, values[:, j])
        finite = values[:, j][np.isfinite(values[:, j])]
        large_nonnegative = kind == "nonnegative" and len(finite) and float(np.nanmax(finite)) > 10.0
        flags.append(kind == "amount" or large_nonnegative)
    return np.array(flags, dtype=bool)


def _transform_values(values: np.ndarray, log_flags: np.ndarray) -> np.ndarray:
    transformed = values.copy()
    transformed[:, log_flags] = np.log1p(np.maximum(transformed[:, log_flags], 0.0))
    return transformed


def _inverse_transform_values(values: np.ndarray, log_flags: np.ndarray) -> np.ndarray:
    restored = values.copy()
    restored[:, log_flags] = np.expm1(restored[:, log_flags])
    return restored


def _raw_error_metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {"original_mae": 0.0, "original_rmse": 0.0}
    err = prediction[mask] - truth[mask]
    return {
        "original_mae": float(np.mean(np.abs(err))),
        "original_rmse": float(np.sqrt(np.mean(err * err))),
    }


def _write_completed_workbook(data: WorkbookData, completed: np.ndarray, output_path: Path) -> None:
    frame = data.metadata.copy()
    completed_frame = pd.DataFrame(completed, columns=data.features)
    out = pd.concat([frame.reset_index(drop=True), completed_frame], axis=1)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")


def _write_feature_dictionary(data: WorkbookData, output_path: Path) -> None:
    rows = []
    for j, feature in enumerate(data.features):
        col = data.values[:, j]
        finite = col[np.isfinite(col)]
        lo, hi = feature_bounds(feature, col)
        rows.append(
            {
                "feature": feature,
                "kind": feature_kind(feature, col),
                "observed_cells": int(np.isfinite(col).sum()),
                "missing_cells": int((~np.isfinite(col)).sum()),
                "lower_bound": lo,
                "upper_bound": hi,
                "closure_group": closure_group_for_feature(feature, data.features, data.values),
                "observed_min": float(np.nanmin(finite)) if len(finite) else None,
                "observed_max": float(np.nanmax(finite)) if len(finite) else None,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def _audit_queue(
    data: WorkbookData,
    candidate: np.ndarray,
    pre_projection: np.ndarray,
    uncertainty_norm: np.ndarray,
    feature_scale: np.ndarray,
    budget: int,
) -> pd.DataFrame:
    missing = ~data.observed
    entries = np.argwhere(missing)
    if len(entries) == 0:
        return pd.DataFrame()

    delta = np.abs(candidate - pre_projection)
    missing_rate = missing.mean(axis=0)
    score = (
        uncertainty_norm * feature_scale.reshape(1, -1)
        + delta / np.maximum(feature_scale.reshape(1, -1), 1e-9)
        + 0.25 * missing_rate.reshape(1, -1)
    )
    order = np.argsort(score[missing])[::-1][:budget]
    selected = entries[order]
    median_unc = float(np.nanmedian(uncertainty_norm[missing]))
    median_delta = float(np.nanmedian(delta[missing]))

    rows = []
    for r, c in selected:
        meta = data.metadata.iloc[int(r)]
        feature = data.features[int(c)]
        group = closure_group_for_feature(feature, data.features, data.values)
        reasons = []
        if group:
            reasons.append("closure-sensitive field")
        if uncertainty_norm[r, c] > median_unc:
            reasons.append("high expert disagreement")
        if delta[r, c] > median_delta:
            reasons.append("projection adjustment")
        if not reasons:
            reasons.append("sparse reported field")
        rows.append(
            {
                "rank": len(rows) + 1,
                "city_code": meta.get("city_code", ""),
                "city_name": meta.get("city_name", ""),
                "country_name": meta.get("country_name", ""),
                "feature": feature,
                "raw_status": "missing",
                "candidate_value": float(candidate[r, c]),
                "projection_delta": float(candidate[r, c] - pre_projection[r, c]),
                "uncertainty_score": float(uncertainty_norm[r, c]),
                "constraint_group": group,
                "why_selected": "; ".join(reasons),
                "suggested_reviewer_action": "verify source value before publication",
            }
        )
    return pd.DataFrame(rows)


def _prediction_records(
    data: WorkbookData,
    prediction: np.ndarray,
    test_mask: np.ndarray,
    output_path: Path,
) -> None:
    rows = []
    for r, c in np.argwhere(test_mask):
        meta = data.metadata.iloc[int(r)]
        true_value = float(data.values[r, c])
        pred_value = float(prediction[r, c])
        rows.append(
            {
                "city_code": meta.get("city_code", ""),
                "city_name": meta.get("city_name", ""),
                "country_name": meta.get("country_name", ""),
                "feature": data.features[int(c)],
                "true_value": true_value,
                "predicted_value": pred_value,
                "absolute_error": abs(pred_value - true_value),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def run_main_experiment(root: Path, config: dict) -> PipelineResult:
    data_path = root / config["data_file"]
    output_dir = root / config.get("output_dir", "outputs/main")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    data = load_city_workbook(
        data_path,
        int(config.get("min_observed_per_feature", 8)),
        list(config.get("feature_prefixes", [])) or None,
    )
    log_flags = _log_transform_features(data.features, data.values)
    model_values = _transform_values(data.values, log_flags)
    keys = group_keys(data.metadata)

    observed = data.observed.copy()
    test_mask = _sample_mask(observed, float(config.get("test_fraction", 0.2)), rng)
    training_pool = observed & ~test_mask
    validation_mask = _sample_mask(training_pool, float(config.get("validation_fraction", 0.15)), rng)
    fit_mask = training_pool & ~validation_mask

    eval_model = fit_pccm(
        model_values,
        fit_mask,
        validation_mask,
        keys,
        int(config.get("svd_rank", 6)),
        int(config.get("svd_iterations", 25)),
        int(config.get("knn_neighbors", 8)),
    )
    eval_model_units = denormalize(eval_model.prediction_norm, eval_model.feature_mean, eval_model.feature_scale)
    eval_raw = _inverse_transform_values(eval_model_units, log_flags)
    eval_projected = project_workbook(eval_raw, data.features, data.values)
    eval_projected_model_units = _transform_values(eval_projected, log_flags)
    metric_scale = np.maximum(eval_model.feature_scale, 0.05)
    metrics = masked_errors(eval_projected_model_units, model_values, test_mask, metric_scale)
    metrics.update(_raw_error_metrics(eval_projected, data.values, test_mask))
    metrics["closure_violation"] = closure_violation(eval_projected, data.features, data.values)
    metrics["test_fraction"] = float(config.get("test_fraction", 0.2))
    metrics["validation_fraction"] = float(config.get("validation_fraction", 0.15))
    metrics["seed"] = seed
    metrics["features"] = len(data.features)
    metrics["cities"] = int(data.values.shape[0])
    metrics["observed_cells"] = int(observed.sum())
    metrics["missing_cells"] = int((~observed).sum())
    metrics["expert_weights"] = eval_model.weights

    delivery_validation = _sample_mask(observed, float(config.get("validation_fraction", 0.15)), rng)
    delivery_fit = observed & ~delivery_validation
    delivery_model = fit_pccm(
        model_values,
        delivery_fit,
        delivery_validation,
        keys,
        int(config.get("svd_rank", 6)),
        int(config.get("svd_iterations", 25)),
        int(config.get("knn_neighbors", 8)),
    )
    delivery_model_units = denormalize(delivery_model.prediction_norm, delivery_model.feature_mean, delivery_model.feature_scale)
    delivery_raw = _inverse_transform_values(delivery_model_units, log_flags)
    delivery_projected = project_workbook(delivery_raw, data.features, data.values)
    completed = data.values.copy()
    completed[~observed] = delivery_projected[~observed]

    _write_completed_workbook(data, completed, output_dir / "completed_workbook.csv")
    _write_feature_dictionary(data, output_dir / "feature_dictionary.csv")
    _prediction_records(data, eval_projected, test_mask, output_dir / "holdout_predictions.csv")
    queue = _audit_queue(
        data,
        delivery_projected,
        delivery_raw,
        delivery_model.uncertainty_norm,
        delivery_model.feature_scale,
        int(config.get("audit_budget", 120)),
    )
    queue.to_csv(output_dir / "audit_queue.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "main_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    summary = [
        "ECO-Waste-PCCM main experiment",
        f"Data file: {config['data_file']}",
        f"Cities: {metrics['cities']}",
        f"Features: {metrics['features']}",
        f"Evaluated held-out cells: {metrics['evaluated_cells']}",
        f"Normalized MAE: {metrics['normalized_mae']:.6f}",
        f"Normalized RMSE: {metrics['normalized_rmse']:.6f}",
        f"Closure violation: {metrics['closure_violation']:.6f}",
        "",
        "Outputs:",
        "- completed_workbook.csv: workbook with reported cells preserved and missing cells completed",
        "- audit_queue.csv: ranked missing cells for human verification",
        "- holdout_predictions.csv: split-safe main-experiment holdout predictions",
        "- feature_dictionary.csv: feature type, bounds, and closure metadata",
        "- main_metrics.json: machine-readable run summary",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    return PipelineResult(output_dir=output_dir, metrics=metrics)
