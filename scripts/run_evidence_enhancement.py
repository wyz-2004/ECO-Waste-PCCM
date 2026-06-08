from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = Path(os.environ["ECOWASTE_PROJECT_ROOT"]).resolve() if os.environ.get("ECOWASTE_PROJECT_ROOT") else SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import ecowaste_core as base
import run_experiment as legacy
from run_aaai_experiments import aggregate, spearman_correlation


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
EPS = 1e-9
HASH_BINS = 24
TEXT_COLUMNS = [
    "method_of_measurement",
    "additional_explanation_for_methof_of_data_collection",
    "point_of_measuremnet",
    "source",
    "weblink",
    "notes",
    "page_figure",
    "date_of_measurement",
]


def stable_hash(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value % HASH_BINS, 1.0 if ((value >> 8) & 1) else -1.0


def text_vector(values: list[object]) -> np.ndarray:
    text = " ".join(base.normalize_text(value).lower() for value in values)
    tokens = re.findall(r"[a-z0-9]{2,}", text)
    vector = np.zeros(HASH_BINS + 5, dtype=float)
    for token in tokens:
        idx, sign = stable_hash(token)
        vector[idx] += sign
    norm = float(np.linalg.norm(vector[:HASH_BINS]))
    if norm > EPS:
        vector[:HASH_BINS] /= norm
    vector[HASH_BINS:] = [
        np.log1p(len(text)) / 8.0,
        np.log1p(len(tokens)) / 6.0,
        float("http" in text),
        float(any(year in text for year in [str(y) for y in range(2018, 2027)])),
        float(bool(text.strip())),
    ]
    return vector


def build_text_tensor(data: base.PreparedData) -> tuple[np.ndarray, list[str]]:
    tensor = np.zeros((len(data.city), len(data.features), HASH_BINS + 5), dtype=np.float32)
    cb = data.codebook.copy()
    available = [column for column in TEXT_COLUMNS if column in cb.columns]
    if not {"city_code", "measurement"}.issubset(cb.columns):
        names = [f"text_hash_{idx:02d}" for idx in range(HASH_BINS)] + [
            "text_length",
            "token_count",
            "has_url_text",
            "recent_year_text",
            "has_any_text",
        ]
        return tensor, names
    grouped = cb.groupby(["city_code", "measurement"], dropna=False)
    lookup: dict[tuple[object, object], np.ndarray] = {}
    for key, frame in grouped:
        values: list[object] = []
        for column in available:
            values.extend(frame[column].tolist())
        lookup[key] = text_vector(values)
    city_codes = data.city["city_code"].tolist()
    for j, feature in enumerate(data.features):
        measurement = feature.split("::", 1)[0]
        for i, city_code in enumerate(city_codes):
            vector = lookup.get((city_code, measurement))
            if vector is not None:
                tensor[i, j, :] = vector
    names = [f"text_hash_{idx:02d}" for idx in range(HASH_BINS)] + [
        "text_length",
        "token_count",
        "has_url_text",
        "recent_year_text",
        "has_any_text",
    ]
    return tensor, names


def provenance_context(data: base.PreparedData) -> tuple[np.ndarray, list[str]]:
    raw = np.nan_to_num(data.evidence_features, nan=0.0)
    static = np.clip(np.nan_to_num(data.static_reliability, nan=0.0), 0.0, 1.0)
    metadata_present = raw[:, :, -4:].mean(axis=2)
    row_mean = np.mean(static, axis=1, keepdims=True)
    row_std = np.std(static, axis=1, keepdims=True)
    col_mean = np.mean(static, axis=0, keepdims=True)
    col_std = np.std(static, axis=0, keepdims=True)
    row_complete = np.mean(metadata_present, axis=1, keepdims=True)
    col_complete = np.mean(metadata_present, axis=0, keepdims=True)
    shape = static.shape
    context = np.stack(
        [
            static,
            np.broadcast_to(row_mean, shape),
            np.broadcast_to(row_std, shape),
            np.broadcast_to(col_mean, shape),
            np.broadcast_to(col_std, shape),
            np.broadcast_to(row_complete, shape),
            np.broadcast_to(col_complete, shape),
            static - np.broadcast_to(row_mean, shape),
            static - np.broadcast_to(col_mean, shape),
        ],
        axis=2,
    )
    names = [
        "rule_reliability",
        "city_provenance_mean",
        "city_provenance_dispersion",
        "field_provenance_mean",
        "field_provenance_dispersion",
        "city_metadata_completeness",
        "field_metadata_completeness",
        "cell_minus_city_reliability",
        "cell_minus_field_reliability",
    ]
    return context, names


def group_hash(data: base.PreparedData, bins: int = 16) -> tuple[np.ndarray, list[str]]:
    matrix = np.zeros((len(data.features), bins), dtype=float)
    for j, feature in enumerate(data.features):
        idx, _ = stable_hash("group:" + base.group_id(feature))
        matrix[j, idx % bins] = 1.0
    return matrix, [f"group_hash_{idx:02d}" for idx in range(bins)]


def masks_for_seed(data: base.PreparedData, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = base.make_random_mask(data, 0.20, seed + 6100)
    pool = data.observed_mask & ~test
    rng = np.random.default_rng(seed + 6101)
    calibration = pool & (rng.random(pool.shape) < 0.18)
    train = pool & ~calibration
    return train, calibration, test


def structural_tensor(
    data: base.PreparedData, train: np.ndarray, prediction: np.ndarray, uncertainty: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    median = legacy.stat_imputer(data, train, "median")
    row_gap = 1.0 - train.mean(axis=1)
    col_gap = 1.0 - train.mean(axis=0)
    constraint = np.array(
        [1.0 if base.group_id(feature) in {"composition", "treatment", "collection"} else 0.0 for feature in data.features]
    )
    groups, group_names = group_hash(data)
    shape = train.shape
    core = np.stack(
        [
            np.log1p(np.maximum(uncertainty, 0.0)),
            np.abs(prediction - median),
            np.broadcast_to(row_gap[:, None], shape),
            np.broadcast_to(col_gap[None, :], shape),
            np.broadcast_to(constraint[None, :], shape),
        ],
        axis=2,
    )
    group_tensor = np.broadcast_to(groups[None, :, :], (shape[0], shape[1], groups.shape[1]))
    return np.concatenate([core, group_tensor], axis=2), [
        "model_uncertainty",
        "expert_disagreement",
        "city_missingness",
        "field_missingness",
        "constraint_sensitivity",
        *group_names,
    ]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.01
    try:
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(design.T @ design + penalty) @ (design.T @ y)
    return coef, mean, scale


def ridge_predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coef, mean, scale = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coef


def choose_ridge(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    tune = rng.random(len(y)) < 0.28
    if tune.sum() < 100 or (~tune).sum() < 200:
        return 10.0
    target = np.log1p(np.maximum(y, 0.0))
    best_alpha, best_corr = 10.0, -np.inf
    for alpha in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        score = ridge_predict(fit_ridge(x[~tune], target[~tune], alpha), x[tune])
        corr = spearman_correlation(score, y[tune])
        if corr > best_corr:
            best_alpha, best_corr = alpha, corr
    return best_alpha


def shuffle_within_field(entries: np.ndarray, values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    for field in np.unique(entries[:, 1]):
        idx = np.flatnonzero(entries[:, 1] == field)
        shuffled[idx] = values[rng.permutation(idx)]
    return shuffled


def adapter_features(metadata: np.ndarray, structural: np.ndarray) -> np.ndarray:
    """Encode metadata as a conditional residual-risk adapter."""
    uncertainty = structural[:, [0]]
    disagreement = structural[:, [1]]
    return np.column_stack(
        [
            metadata,
            metadata * uncertainty,
            metadata * disagreement,
            uncertainty * disagreement,
        ]
    )


def fit_residual_adapter(
    x_base: np.ndarray,
    x_metadata: np.ndarray,
    error: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], float, np.ndarray]:
    rng = np.random.default_rng(seed)
    tune = rng.random(len(error)) < 0.30
    if tune.sum() < 150 or (~tune).sum() < 300:
        tune = np.arange(len(error)) % 4 == 0
    target = np.log1p(error)
    base_alpha = choose_ridge(x_base[~tune], error[~tune], seed + 1)
    base_model = fit_ridge(x_base[~tune], target[~tune], base_alpha)
    base_fit = ridge_predict(base_model, x_base[~tune])
    base_tune = ridge_predict(base_model, x_base[tune])
    adapter_x = adapter_features(x_metadata, x_base)
    residual_fit = target[~tune] - base_fit
    best_model = fit_ridge(adapter_x[~tune], residual_fit, 30.0)
    best_corr = -np.inf
    for alpha in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        candidate = fit_ridge(adapter_x[~tune], residual_fit, alpha)
        residual_score = ridge_predict(candidate, adapter_x[tune])
        corr = spearman_correlation(residual_score, target[tune] - base_tune)
        if corr > best_corr:
            best_corr = corr
            best_model = candidate
    residual_tune = ridge_predict(best_model, adapter_x[tune])
    best_lambda = 0.0
    best_utility = spearman_correlation(base_tune, error[tune]) + 0.20 * pairwise_auc(base_tune, error[tune])
    for strength in [0.10, 0.25, 0.50, 0.75, 1.0]:
        combined = base_tune + strength * residual_tune
        utility = spearman_correlation(combined, error[tune]) + 0.20 * pairwise_auc(combined, error[tune])
        if utility > best_utility + 0.002:
            best_utility = utility
            best_lambda = strength
    return base_model, best_model, best_lambda, tune


def pairwise_auc(score: np.ndarray, error: np.ndarray) -> float:
    threshold = np.quantile(error, 0.75)
    positive = score[error >= threshold]
    negative = score[error < threshold]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    rng = np.random.default_rng(20260607)
    size = min(5000, len(positive) * len(negative))
    pos = positive[rng.integers(0, len(positive), size=size)]
    neg = negative[rng.integers(0, len(negative), size=size)]
    return float(np.mean(pos > neg) + 0.5 * np.mean(pos == neg))


def metric_row(
    variant: str,
    seed: int,
    score: np.ndarray,
    error: np.ndarray,
    base_score: np.ndarray,
) -> dict[str, object]:
    low, high = np.quantile(score, [0.25, 0.75])
    top = float(error[score >= high].mean())
    bottom = float(error[score <= low].mean())
    order = np.argsort(score)[::-1]
    budget = max(1, int(round(0.20 * len(order))))
    capture = float(error[order[:budget]].sum() / max(error.sum(), EPS))
    base_design = np.column_stack([np.ones(len(base_score)), base_score])
    base_coef = np.linalg.pinv(base_design) @ error
    error_residual = error - base_design @ base_coef
    score_coef = np.linalg.pinv(base_design) @ score
    score_residual = score - base_design @ score_coef
    return {
        "variant": variant,
        "seed": seed,
        "risk_error_spearman": spearman_correlation(score, error),
        "partial_evidence_spearman": spearman_correlation(score_residual, error_residual),
        "high_to_low_error_ratio": top / max(bottom, EPS),
        "top20_error_capture": capture,
        "high_error_auc": pairwise_auc(score, error),
        "risk_prediction_mae": float(np.mean(np.abs(np.expm1(score) - error))),
    }


def run_experiment(data: base.PreparedData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text, text_names = build_text_tensor(data)
    context, context_names = provenance_context(data)
    metadata = np.concatenate([np.nan_to_num(data.evidence_features, nan=0.0), context, text], axis=2)
    metadata_names = [*data.evidence_feature_names, *context_names, *text_names]
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        train, calibration, test = masks_for_seed(data, seed)
        prediction, uncertainty = legacy.safe_pccm_predict(data, train, seed=seed + 67, use_evidence=False)
        structural, structural_names = structural_tensor(data, train, prediction, uncertainty)
        cal_entries = np.argwhere(calibration)
        test_entries = np.argwhere(test)
        x_base_cal = structural[calibration]
        x_base_test = structural[test]
        x_ev_cal = metadata[calibration]
        x_ev_test = metadata[test]
        cal_error = np.abs(prediction[calibration] - data.model_values[calibration])
        test_error = np.abs(prediction[test] - data.model_values[test])
        base_model, adapter_model, strength, tune = fit_residual_adapter(x_base_cal, x_ev_cal, cal_error, seed + 1)
        base_cal = ridge_predict(base_model, x_base_cal)
        base_test = ridge_predict(base_model, x_base_test)
        residual_cal = ridge_predict(adapter_model, adapter_features(x_ev_cal, x_base_cal))
        residual_test = ridge_predict(adapter_model, adapter_features(x_ev_test, x_base_test))
        enhanced_test = base_test + strength * residual_test

        shuffled_cal = shuffle_within_field(cal_entries, x_ev_cal, seed + 3)
        shuffled_test = shuffle_within_field(test_entries, x_ev_test, seed + 4)
        shuffled_base_model, shuffled_adapter_model, shuffled_strength, _ = fit_residual_adapter(
            x_base_cal, shuffled_cal, cal_error, seed + 5
        )
        shuffled_base_test = ridge_predict(shuffled_base_model, x_base_test)
        shuffled_residual_test = ridge_predict(shuffled_adapter_model, adapter_features(shuffled_test, x_base_test))
        shuffled_score = shuffled_base_test + shuffled_strength * shuffled_residual_test

        raw_gap = 1.0 - np.clip(data.reliability[test], 0.0, 1.0)
        rows.extend(
            [
                metric_row("Structural no-evidence", seed, base_test, test_error, base_test),
                metric_row("Raw learned evidence", seed, raw_gap, test_error, base_test),
                metric_row("Within-field shuffled residual adapter", seed, shuffled_residual_test, test_error, shuffled_base_test),
                metric_row("Ours evidence residual signal", seed, residual_test, test_error, base_test),
                metric_row("Ours safe evidence adapter", seed, enhanced_test, test_error, base_test),
            ]
        )

        base_capture = metric_row("base", seed, base_cal[tune], cal_error[tune], base_cal[tune])["top20_error_capture"]
        full_cal_score = base_cal[tune] + strength * residual_cal[tune]
        full_capture = metric_row("full", seed, full_cal_score, cal_error[tune], base_cal[tune])["top20_error_capture"]
        activation_rows.append(
            {
                "seed": seed,
                "base_validation_capture": base_capture,
                "evidence_validation_capture": full_capture,
                "adapter_strength": strength,
                "activate_evidence": float(strength > 0),
                "test_capture_gain": metric_row("full", seed, enhanced_test, test_error, base_test)["top20_error_capture"]
                - metric_row("base", seed, base_test, test_error, base_test)["top20_error_capture"],
                "test_spearman_gain": metric_row("full", seed, enhanced_test, test_error, base_test)["risk_error_spearman"]
                - metric_row("base", seed, base_test, test_error, base_test)["risk_error_spearman"],
            }
        )

        coef = adapter_model[0][1:]
        names = [
            *metadata_names,
            *[f"{name}_x_uncertainty" for name in metadata_names],
            *[f"{name}_x_disagreement" for name in metadata_names],
            "uncertainty_x_disagreement",
        ]
        for name, value in zip(names, coef):
            weight_rows.append({"seed": seed, "feature": name, "coefficient": float(value), "block": "evidence_adapter"})
    return pd.DataFrame(rows), pd.DataFrame(weight_rows), pd.DataFrame(activation_rows)


def paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    pivot = raw.pivot(index="seed", columns="variant")
    ours = "Ours safe evidence adapter"
    rows = []
    for baseline in ["Structural no-evidence", "Raw learned evidence", "Within-field shuffled residual adapter"]:
        for metric in ["risk_error_spearman", "partial_evidence_spearman", "top20_error_capture", "high_error_auc"]:
            diff = pivot[metric][ours] - pivot[metric][baseline]
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "paired_gain": float(diff.mean()),
                    "wins": int((diff > 1e-10).sum()),
                    "ties": int((np.abs(diff) <= 1e-10).sum()),
                    "losses": int((diff < -1e-10).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)
    raw, weights, activation = run_experiment(data)
    summary = aggregate(raw, ["variant"])
    effects = paired_effects(raw)
    raw.to_csv(TABLE_DIR / "evidence_encoder_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "evidence_encoder_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(TABLE_DIR / "evidence_encoder_paired_effects.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(TABLE_DIR / "evidence_encoder_weights.csv", index=False, encoding="utf-8-sig")
    activation.to_csv(TABLE_DIR / "evidence_encoder_activation_by_seed.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "encoder": "cross-fitted residualized provenance encoder",
        "metadata_blocks": ["structured codebook fields", "hashed source text", "city/field provenance consistency"],
        "counterfactual": "within-field shuffled metadata",
        "seeds": SEEDS,
    }
    (OUT_DIR / "evidence_encoder_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
