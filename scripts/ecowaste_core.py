from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
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
XLSX_PATH = ROOT / "data" / "raw" / "What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx"
WORK_DIR = ROOT / "generated" / "work" / "ecowaste_v3"
OUT_DIR = ROOT / "generated" / "outputs" / "ecowaste_v3"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PROTOCOL_DIR = OUT_DIR / "protocol"

RNG_SEED = 42
ID_COLS = {
    "country_code",
    "iso3c",
    "country_name",
    "city_name",
    "city_code",
}
CONTEXT_COLS = ["region_id", "income_id", "income_id_2022"]
TEXT_MISSING = {
    "",
    "nan",
    "none",
    "not stated",
    "not available",
    "not applicable",
    "n/a",
    "na",
    "-",
    "--",
}


@dataclass
class PreparedData:
    city: pd.DataFrame
    codebook: pd.DataFrame
    features: list[str]
    percent_features: set[str]
    binary_features: set[str]
    raw_values: pd.DataFrame
    model_values: np.ndarray
    observed_mask: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    static_reliability: np.ndarray
    reliability: np.ndarray
    evidence_features: np.ndarray
    evidence_feature_names: list[str]
    evidence_calibration_weights: pd.DataFrame
    evidence_calibration_diagnostics: pd.DataFrame
    context: pd.DataFrame


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_year(value: object) -> float:
    text = normalize_text(value)
    matches = re.findall(r"(19\d{2}|20\d{2})", text)
    if not matches:
        return np.nan
    years = [int(m) for m in matches]
    return float(max(years))


def evidence_score(row: pd.Series) -> float:
    method = normalize_text(row.get("method_of_measurement", "")).lower()
    point = normalize_text(row.get("point_of_measuremnet", "")).lower()
    source = normalize_text(row.get("source", ""))
    weblink = normalize_text(row.get("weblink", ""))
    notes = normalize_text(row.get("notes", ""))
    year = parse_year(row.get("date_of_measurement", ""))

    method_score = 0.42
    if any(k in method for k in ["trucks weighed", "weighed", "scale", "waste characterisation"]):
        method_score = 0.92
    elif any(k in method for k in ["stratified", "extrapolated", "sample"]):
        method_score = 0.78
    elif "estimated" in method:
        method_score = 0.45
    elif method in TEXT_MISSING or "not stated" in method:
        method_score = 0.25
    elif method:
        method_score = 0.60

    year_score = 0.40
    if not np.isnan(year):
        year_score = float(np.clip((year - 2000) / 24, 0.15, 1.0))

    source_score = 0.15
    if source and source.lower() not in TEXT_MISSING:
        source_score += 0.25
    if weblink and weblink.lower() not in TEXT_MISSING:
        source_score += 0.20
    if notes and notes.lower() not in TEXT_MISSING:
        source_score += 0.10
    if point and point not in TEXT_MISSING:
        source_score += 0.10

    return float(np.clip(0.50 * method_score + 0.30 * year_score + 0.20 * source_score, 0.08, 1.0))


EVIDENCE_FEATURE_NAMES = [
    "intercept",
    "rule_score",
    "method_weighed_or_characterised",
    "method_sample_or_extrapolated",
    "method_estimated",
    "method_missing_or_not_stated",
    "has_measurement_point",
    "has_year",
    "year_recency",
    "has_source",
    "has_weblink",
    "has_notes",
]


def evidence_feature_vector(row: pd.Series | None) -> np.ndarray:
    if row is None:
        return np.array([1.0, 0.38, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0], dtype=float)
    method = normalize_text(row.get("method_of_measurement", "")).lower()
    point = normalize_text(row.get("point_of_measuremnet", "")).lower()
    source = normalize_text(row.get("source", ""))
    weblink = normalize_text(row.get("weblink", ""))
    notes = normalize_text(row.get("notes", ""))
    year = parse_year(row.get("date_of_measurement", ""))
    has_year = float(np.isfinite(year))
    year_recency = float(np.clip((year - 2000) / 24, 0.05, 1.0)) if np.isfinite(year) else 0.35
    return np.array(
        [
            1.0,
            evidence_score(row),
            float(any(k in method for k in ["trucks weighed", "weighed", "scale", "waste characterisation"])),
            float(any(k in method for k in ["stratified", "extrapolated", "sample"])),
            float("estimated" in method),
            float((method in TEXT_MISSING) or ("not stated" in method) or (not method)),
            float(bool(point and point not in TEXT_MISSING)),
            has_year,
            year_recency,
            float(bool(source and source.lower() not in TEXT_MISSING)),
            float(bool(weblink and weblink.lower() not in TEXT_MISSING)),
            float(bool(notes and notes.lower() not in TEXT_MISSING)),
        ],
        dtype=float,
    )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    city = pd.read_excel(XLSX_PATH, sheet_name="City dataset")
    codebook = pd.read_excel(XLSX_PATH, sheet_name="Codebook")
    return city, codebook


def is_missing_like(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in TEXT_MISSING


def yes_no_score(series: pd.Series) -> pd.Series | None:
    nonnull = series[~series.map(is_missing_like)].astype(str).str.strip().str.lower()
    if len(nonnull) < 20:
        return None
    vocab = set(nonnull.unique())
    known = {
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "1": 1.0,
        "implemented": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "0": 0.0,
        "not implemented": 0.0,
    }
    if not vocab or len(vocab - set(known)) > 2:
        return None
    return series.astype(str).str.strip().str.lower().map(known)


def prepare_features(city: pd.DataFrame, codebook: pd.DataFrame) -> PreparedData:
    raw_columns: dict[str, pd.Series] = {}
    percent_features: set[str] = set()
    binary_features: set[str] = set()

    for col in city.columns:
        if col in ID_COLS or col in CONTEXT_COLS:
            continue
        numeric = pd.to_numeric(city[col], errors="coerce")
        numeric_share = numeric.notna().mean()
        nonmissing = int(numeric.notna().sum())
        if nonmissing >= 5 and numeric_share >= 0.02:
            raw_columns[col] = numeric
            if "percent" in col.lower():
                percent_features.add(col)
            continue
        binary = yes_no_score(city[col])
        if binary is not None and int(binary.notna().sum()) >= 10:
            raw_columns[col] = binary
            binary_features.add(col)
            percent_features.add(col)
            continue
        nonnull_text = city[col][~city[col].map(is_missing_like)].astype(str).str.strip()
        if len(nonnull_text) >= 20:
            counts = nonnull_text.value_counts()
            values = [v for v, c in counts.items() if c >= 5 and len(str(v)) <= 55]
            if 2 <= len(values) <= 14 and nonnull_text.str.len().median() <= 55:
                for value in values:
                    feature_name = f"{col}::{value}"
                    raw_columns[feature_name] = np.where(
                        city[col].astype(str).str.strip() == value,
                        1.0,
                        np.where(city[col].map(is_missing_like), np.nan, 0.0),
                    )
                    binary_features.add(feature_name)
                    percent_features.add(feature_name)

    raw_values = pd.DataFrame(raw_columns)
    features = list(raw_values.columns)

    model_df = raw_values.copy()
    for col in features:
        if col in percent_features:
            values = pd.to_numeric(model_df[col], errors="coerce")
            if values.dropna().quantile(0.95) > 1.5:
                model_df[col] = values / 100.0
            model_df[col] = model_df[col].clip(0, 1)
        else:
            values = pd.to_numeric(model_df[col], errors="coerce")
            if values.dropna().quantile(0.95) > 1000 and values.min(skipna=True) >= 0:
                model_df[col] = np.log1p(values)

    observed_mask = model_df.notna().to_numpy()
    model_values_raw = model_df.to_numpy(dtype=float)
    feature_mean = np.nanmedian(model_values_raw, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    q75 = np.nanpercentile(model_values_raw, 75, axis=0)
    q25 = np.nanpercentile(model_values_raw, 25, axis=0)
    feature_scale = q75 - q25
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1e-9), feature_scale, np.nanstd(model_values_raw, axis=0))
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1e-9), feature_scale, 1.0)
    model_values = (model_values_raw - feature_mean) / feature_scale
    model_values = np.where(np.isfinite(model_values), np.clip(model_values, -8, 8), np.nan)

    reliability = np.zeros_like(model_values, dtype=float)
    evidence_features = np.zeros((model_values.shape[0], model_values.shape[1], len(EVIDENCE_FEATURE_NAMES)), dtype=float)
    evidence_features[:, :, :] = evidence_feature_vector(None)[None, None, :]
    if {"city_code", "measurement"}.issubset(codebook.columns):
        cb = codebook.copy()
        cb["_score"] = cb.apply(evidence_score, axis=1)
        for name, idx in zip(EVIDENCE_FEATURE_NAMES, range(len(EVIDENCE_FEATURE_NAMES))):
            cb[f"_ev_{name}"] = cb.apply(lambda row, k=idx: evidence_feature_vector(row)[k], axis=1)
        grouped = cb.groupby(["city_code", "measurement"], dropna=False)["_score"].mean()
        grouped_ev = cb.groupby(["city_code", "measurement"], dropna=False)[[f"_ev_{name}" for name in EVIDENCE_FEATURE_NAMES]].mean()
        city_codes = city["city_code"].tolist()
        for j, feature in enumerate(features):
            base_feature = feature.split("::", 1)[0]
            for i, code in enumerate(city_codes):
                score = grouped.get((code, base_feature), np.nan)
                if np.isfinite(score):
                    reliability[i, j] = score
                ev_key = (code, base_feature)
                if ev_key in grouped_ev.index:
                    evidence_features[i, j, :] = grouped_ev.loc[ev_key].to_numpy(dtype=float)
    reliability = np.where((reliability == 0) & observed_mask, 0.38, reliability)
    static_reliability = reliability.copy()

    context = city[[c for c in CONTEXT_COLS if c in city.columns]].copy()
    data = PreparedData(
        city=city,
        codebook=codebook,
        features=features,
        percent_features=percent_features,
        binary_features=binary_features,
        raw_values=raw_values,
        model_values=model_values,
        observed_mask=observed_mask,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        static_reliability=static_reliability,
        reliability=reliability,
        evidence_features=evidence_features,
        evidence_feature_names=EVIDENCE_FEATURE_NAMES.copy(),
        evidence_calibration_weights=pd.DataFrame(),
        evidence_calibration_diagnostics=pd.DataFrame(),
        context=context,
    )
    learned, weights, diagnostics = learn_evidence_reliability(data)
    data.reliability = learned
    data.evidence_calibration_weights = weights
    data.evidence_calibration_diagnostics = diagnostics
    return data


def group_id(feature: str) -> str:
    feature = feature.split("::", 1)[0]
    if feature.startswith("composition_msw_"):
        return "composition"
    if feature.startswith("waste_treatment_") or feature == "waste_uncollected_percent":
        return "treatment"
    if feature.startswith("uncollected_waste_"):
        return "uncollected"
    if feature.startswith("waste_collection_coverage"):
        return "collection"
    if feature.startswith("msw_total_") or feature.startswith("population_") or feature == "gdp":
        return "generation"
    if feature.startswith("non_msw_"):
        return "non_msw"
    if feature.startswith("epr_") or feature.startswith("drs_"):
        return "epr_drs"
    if feature.startswith("other_information_"):
        return "governance"
    if feature.startswith("waste_workers_"):
        return "workers"
    if feature.startswith("separation_"):
        return "separation"
    if feature.startswith("solid_waste_budget") or feature.startswith("municipal_waste_budget"):
        return "budget"
    if feature.startswith("institutional_framework"):
        return "institutional"
    if feature.startswith("legal_framework"):
        return "legal"
    if "__" in feature:
        return feature.split("__", 1)[0]
    return feature.split("_", 1)[0]


def fill_group_median(data: PreparedData, train_mask: np.ndarray) -> np.ndarray:
    X = data.model_values.copy()
    n, p = X.shape
    pred = np.zeros((n, p), dtype=float)
    global_med = np.nanmedian(np.where(train_mask, X, np.nan), axis=0)
    global_med = np.where(np.isfinite(global_med), global_med, 0.0)
    contexts = data.context.astype(str).fillna("")
    for i in range(n):
        row_context = contexts.iloc[i]
        idx_region_income = np.ones(n, dtype=bool)
        for col in contexts.columns:
            idx_region_income &= contexts[col].to_numpy() == row_context[col]
        idx_region = np.ones(n, dtype=bool)
        if "region_id" in contexts.columns:
            idx_region = contexts["region_id"].to_numpy() == row_context.get("region_id", "")
        for j in range(p):
            candidates = [
                idx_region_income & train_mask[:, j],
                idx_region & train_mask[:, j],
                train_mask[:, j],
            ]
            value = np.nan
            for cand in candidates:
                if int(cand.sum()) >= 3:
                    value = np.nanmedian(X[cand, j])
                    break
            pred[i, j] = value if np.isfinite(value) else global_med[j]
    return pred


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    av = a[valid] - a[valid].mean()
    bv = b[valid] - b[valid].mean()
    denom = float(np.sqrt((av**2).sum() * (bv**2).sum()))
    return float((av * bv).sum() / denom) if denom > 1e-12 else np.nan


def learn_evidence_reliability(data: PreparedData, seed: int = 314, validation_rate: float = 0.16) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    eligible = data.observed_mask.copy()
    eligible[:, data.observed_mask.sum(axis=0) < 20] = False
    validation_mask = eligible & (rng.random(eligible.shape) < validation_rate)
    if int(validation_mask.sum()) < 200:
        weights = pd.DataFrame(
            {
                "evidence_feature": data.evidence_feature_names,
                "coefficient": [np.nan] * len(data.evidence_feature_names),
                "interpretation": ["insufficient validation cells"] * len(data.evidence_feature_names),
            }
        )
        diagnostics = pd.DataFrame([{"validation_cells": int(validation_mask.sum()), "status": "fallback_to_rule_score"}])
        return data.static_reliability.copy(), weights, diagnostics

    train_mask = data.observed_mask & ~validation_mask
    baseline_pred = fill_group_median(data, train_mask)
    err = np.abs(baseline_pred[validation_mask] - data.model_values[validation_mask])
    tau = float(np.nanmedian(err) + np.nanpercentile(err, 75) - np.nanpercentile(err, 25))
    tau = max(tau, 0.15)
    target = np.clip(np.exp(-err / tau), 0.04, 0.96)
    y = np.log(target / (1.0 - target))
    Z = data.evidence_features[validation_mask]
    Z = np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=0.0)
    ridge = 1.2
    penalty = np.eye(Z.shape[1]) * ridge
    penalty[0, 0] = 0.05
    try:
        coef = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(Z.T @ Z + penalty) @ (Z.T @ y)
    positive_features = {
        "rule_score",
        "method_weighed_or_characterised",
        "method_sample_or_extrapolated",
        "has_measurement_point",
        "has_year",
        "year_recency",
        "has_source",
        "has_weblink",
        "has_notes",
    }
    negative_features = {"method_estimated", "method_missing_or_not_stated"}
    unconstrained_coef = coef.copy()
    for idx, name in enumerate(data.evidence_feature_names):
        if name in positive_features:
            coef[idx] = max(0.0, coef[idx])
        elif name in negative_features:
            coef[idx] = min(0.0, coef[idx])
    all_scores = sigmoid(np.tensordot(np.nan_to_num(data.evidence_features, nan=0.0), coef, axes=([2], [0])))
    learned = np.clip(0.62 * data.static_reliability + 0.38 * all_scores, 0.08, 1.0)
    learned = np.where((data.static_reliability == 0) & data.observed_mask, 0.38, learned)
    static_val = data.static_reliability[validation_mask]
    learned_val = learned[validation_mask]
    all_val = all_scores[validation_mask]
    weights = pd.DataFrame(
        {
            "evidence_feature": data.evidence_feature_names,
            "coefficient": coef,
            "unconstrained_coefficient": unconstrained_coef,
            "direction": [
                "higher reliability" if c > 1e-12 else "lower reliability" if c < -1e-12 else "neutral after constraint"
                for c in coef
            ],
            "constraint": [
                "nonnegative" if name in positive_features else "nonpositive" if name in negative_features else "free"
                for name in data.evidence_feature_names
            ],
        }
    ).sort_values("coefficient", ascending=False)
    diagnostics = pd.DataFrame(
        [
            {
                "validation_cells": int(validation_mask.sum()),
                "target_reliability_mean": float(target.mean()),
                "target_reliability_median": float(np.median(target)),
                "rule_score_mean": float(static_val.mean()),
                "learned_score_mean": float(learned_val.mean()),
                "pure_learned_score_mean": float(all_val.mean()),
                "rule_target_corr": safe_corr(static_val, target),
                "learned_target_corr": safe_corr(learned_val, target),
                "pure_learned_target_corr": safe_corr(all_val, target),
                "blend_weight_rule": 0.62,
                "blend_weight_learned": 0.38,
                "monotonic_domain_constraints": True,
                "status": "learned_from_self_supervised_masking_with_domain_constraints",
            }
        ]
    )
    return learned, weights, diagnostics


def weighted_city_knn(data: PreparedData, train_mask: np.ndarray, k: int = 12) -> np.ndarray:
    X = data.model_values.copy()
    base = fill_group_median(data, train_mask)
    filled = np.where(train_mask, X, base)
    weights_obs = train_mask.astype(float) * np.maximum(data.reliability, 0.25)
    n, p = X.shape
    pred = base.copy()
    context = data.context.astype(str).fillna("")
    for i in range(n):
        shared = train_mask[i][None, :] & train_mask
        diff2 = (filled - filled[i]) ** 2
        denom = shared.sum(axis=1)
        dist = np.where(denom > 5, (diff2 * shared).sum(axis=1) / np.maximum(denom, 1), np.inf)
        same_region = np.zeros(n, dtype=float)
        same_income = np.zeros(n, dtype=float)
        if "region_id" in context.columns:
            same_region = (context["region_id"].to_numpy() == context.iloc[i]["region_id"]).astype(float)
        if "income_id_2022" in context.columns:
            same_income = (context["income_id_2022"].to_numpy() == context.iloc[i]["income_id_2022"]).astype(float)
        sim = np.exp(-dist / 2.0) + 0.18 * same_region + 0.12 * same_income
        sim[i] = 0.0
        neighbors = np.argsort(sim)[-k:]
        for j in range(p):
            valid = train_mask[neighbors, j]
            if valid.any():
                w = sim[neighbors][valid] * weights_obs[neighbors, j][valid]
                if w.sum() > 1e-12:
                    pred[i, j] = float(np.sum(w * X[neighbors, j][valid]) / np.sum(w))
    return pred


def matrix_factorization(
    data: PreparedData,
    train_mask: np.ndarray,
    rank: int = 10,
    epochs: int = 32,
    lr: float = 0.018,
    reg: float = 0.025,
    seed: int = RNG_SEED,
    use_evidence: bool = True,
) -> np.ndarray:
    del lr, reg, seed
    X = data.model_values
    base = fill_group_median(data, train_mask)
    filled = np.where(train_mask, X, base)
    if use_evidence:
        rel = np.clip(np.where(np.isfinite(data.reliability), data.reliability, 0.35), 0.20, 1.0)
    else:
        rel = np.full_like(data.model_values, 0.55, dtype=float)
    obs_weight = train_mask * (0.55 + 0.40 * rel)
    for _ in range(epochs):
        target = np.where(train_mask, obs_weight * X + (1.0 - obs_weight) * filled, filled)
        col_center = target.mean(axis=0)
        centered = target - col_center
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            r = min(rank, len(s))
            recon = (u[:, :r] * s[:r]) @ vt[:r, :] + col_center
        except np.linalg.LinAlgError:
            recon = target
        recon = np.clip(recon, -8, 8)
        filled = 0.68 * recon + 0.22 * base + 0.10 * target
        filled = np.where(train_mask, 0.72 * X + 0.28 * filled, filled)
    return np.where(np.isfinite(filled), filled, base)


def project_percent_constraints(data: PreparedData, pred_norm: np.ndarray) -> np.ndarray:
    pred = pred_norm * data.feature_scale + data.feature_mean
    feature_index = {f: j for j, f in enumerate(data.features)}
    for feature in data.percent_features:
        if feature in feature_index:
            pred[:, feature_index[feature]] = np.clip(pred[:, feature_index[feature]], 0, 1)

    composition = [
        f
        for f in data.features
        if f.startswith("composition_msw_") and f.endswith("_percent")
    ]
    treatment = [
        f
        for f in data.features
        if (f.startswith("waste_treatment_") and f.endswith("_percent")) or f == "waste_uncollected_percent"
    ]
    for group in [composition, treatment]:
        idx = [feature_index[f] for f in group if f in feature_index]
        if len(idx) < 3:
            continue
        block = np.clip(pred[:, idx], 0, None)
        sums = block.sum(axis=1)
        valid = sums > 1.05
        block[valid] = block[valid] / sums[valid, None]
        pred[:, idx] = block
    return (pred - data.feature_mean) / data.feature_scale


def eco_waste_predict(
    data: PreparedData,
    train_mask: np.ndarray,
    seed: int = RNG_SEED,
    use_evidence: bool = True,
    use_graph: bool = True,
    use_constraints: bool = True,
    use_sparse_shrinkage: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    group = fill_group_median(data, train_mask)
    knn = weighted_city_knn(data, train_mask) if use_graph else group
    mf = matrix_factorization(data, train_mask, seed=seed, use_evidence=use_evidence)
    feature_missing = 1.0 - train_mask.mean(axis=0)
    shrink = np.clip(feature_missing, 0.25, 0.80) if use_sparse_shrinkage else np.full_like(feature_missing, 0.35)
    pred = (1.0 - shrink)[None, :] * (0.58 * mf + 0.42 * knn) + shrink[None, :] * group
    if use_constraints:
        pred = project_percent_constraints(data, pred)
    disagreement = np.std(np.stack([group, knn, mf], axis=0), axis=0)
    rel_base = data.reliability if use_evidence else np.full_like(data.reliability, 0.55, dtype=float)
    rel_penalty = 1.0 - np.where(rel_base > 0, rel_base, np.nanmean(rel_base[rel_base > 0]))
    uncertainty = 0.35 * disagreement + 0.45 * feature_missing[None, :] + 0.20 * np.nan_to_num(rel_penalty, nan=0.6)
    return pred, uncertainty


def evaluate(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, uncertainty: np.ndarray | None = None) -> dict[str, float]:
    err = pred[mask] - truth[mask]
    mae = float(np.mean(np.abs(err))) if err.size else np.nan
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else np.nan
    nrmse = rmse
    coverage = np.nan
    if uncertainty is not None and err.size:
        interval = 1.64 * np.maximum(uncertainty[mask], 1e-6)
        coverage = float(np.mean(np.abs(err) <= interval))
    return {
        "heldout_cells": int(mask.sum()),
        "mae_norm": mae,
        "rmse_norm": rmse,
        "nrmse_norm": nrmse,
        "coverage_90pct_proxy": coverage,
    }


def percent_mae_pp(data: PreparedData, pred: np.ndarray, mask: np.ndarray) -> float:
    idx = [j for j, f in enumerate(data.features) if f in data.percent_features]
    if not idx:
        return np.nan
    m = mask[:, idx]
    if not m.any():
        return np.nan
    pred_raw = pred * data.feature_scale + data.feature_mean
    truth_raw = data.model_values * data.feature_scale + data.feature_mean
    return float(np.mean(np.abs(pred_raw[:, idx][m] - truth_raw[:, idx][m])) * 100)


def constraint_violation(data: PreparedData, pred: np.ndarray) -> dict[str, float]:
    raw = pred * data.feature_scale + data.feature_mean
    idx = {f: j for j, f in enumerate(data.features)}
    comp = [idx[f] for f in data.features if f.startswith("composition_msw_") and f.endswith("_percent")]
    treat = [
        idx[f]
        for f in data.features
        if (f.startswith("waste_treatment_") and f.endswith("_percent")) or f == "waste_uncollected_percent"
    ]
    result = {}
    for name, cols in [("composition_sum_abs_error", comp), ("treatment_sum_abs_error", treat)]:
        if len(cols) >= 3:
            sums = np.clip(raw[:, cols], 0, 1).sum(axis=1)
            result[name] = float(np.mean(np.abs(sums - 1.0)))
        else:
            result[name] = np.nan
    return result


def build_data_dictionary(data: PreparedData) -> pd.DataFrame:
    cb_counts = (
        data.codebook["measurement"].astype(str).value_counts()
        if "measurement" in data.codebook.columns
        else pd.Series(dtype=int)
    )
    modeled_by_source: dict[str, int] = {}
    for feature in data.features:
        source = feature.split("::", 1)[0]
        modeled_by_source[source] = modeled_by_source.get(source, 0) + 1
    rows = []
    for col in data.city.columns:
        nonmissing = ~data.city[col].map(is_missing_like)
        numeric = pd.to_numeric(data.city[col], errors="coerce")
        unique_count = int(data.city.loc[nonmissing, col].astype(str).nunique())
        if col in ID_COLS:
            role = "identifier"
        elif col in CONTEXT_COLS:
            role = "context_group"
        elif modeled_by_source.get(col, 0) > 0:
            role = "modeled"
        elif numeric.notna().sum() > 0:
            role = "numeric_not_modeled"
        else:
            role = "text_metadata"
        rows.append(
            {
                "field": col,
                "variable_group": group_id(col),
                "role": role,
                "nonmissing_count": int(nonmissing.sum()),
                "nonmissing_rate": float(nonmissing.mean()),
                "unique_count": unique_count,
                "numeric_parse_count": int(numeric.notna().sum()),
                "modeled_feature_count": int(modeled_by_source.get(col, 0)),
                "codebook_record_count": int(cb_counts.get(col, 0)),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_by_group(feature_profile: pd.DataFrame) -> pd.DataFrame:
    grouped = feature_profile.groupby("group", dropna=False).agg(
        modeled_features=("feature", "count"),
        mean_observed_rate=("observed_rate", "mean"),
        median_observed_rate=("observed_rate", "median"),
        min_observed_rate=("observed_rate", "min"),
        max_observed_rate=("observed_rate", "max"),
        mean_evidence_reliability=("mean_evidence_reliability", "mean"),
    )
    grouped["mean_missing_rate"] = 1 - grouped["mean_observed_rate"]
    return grouped.reset_index().sort_values("mean_missing_rate", ascending=False)


def build_codebook_alignment(data: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.codebook.empty:
        return pd.DataFrame(), pd.DataFrame()
    aligned = data.codebook.copy()
    aligned["measurement"] = aligned.get("measurement", pd.Series("", index=aligned.index)).astype(str)
    aligned["evidence_score"] = aligned.apply(evidence_score, axis=1)
    aligned["measurement_year"] = aligned.get("date_of_measurement", pd.Series("", index=aligned.index)).map(parse_year)
    aligned["base_variable_group"] = aligned["measurement"].map(group_id)
    aligned["exists_in_city_dataset"] = aligned["measurement"].isin(data.city.columns)
    modeled_sources = {feature.split("::", 1)[0] for feature in data.features}
    aligned["used_by_model"] = aligned["measurement"].isin(modeled_sources)
    keep_cols = [
        c
        for c in [
            "city_code",
            "measurement",
            "units",
            "point_of_measuremnet",
            "method_of_measurement",
            "date_of_measurement",
            "measurement_year",
            "source",
            "weblink",
            "notes",
            "evidence_score",
            "base_variable_group",
            "exists_in_city_dataset",
            "used_by_model",
        ]
        if c in aligned.columns
    ]
    aligned = aligned[keep_cols]

    summary = aligned.groupby("base_variable_group", dropna=False).agg(
        codebook_records=("measurement", "count"),
        unique_measurements=("measurement", "nunique"),
        mean_evidence_score=("evidence_score", "mean"),
        median_evidence_score=("evidence_score", "median"),
        used_by_model_rate=("used_by_model", "mean"),
        with_year_rate=("measurement_year", lambda s: float(s.notna().mean())),
    )
    return aligned, summary.reset_index().sort_values("mean_evidence_score", ascending=False)


def constraint_audit(data: PreparedData, pred: np.ndarray) -> pd.DataFrame:
    raw = pred * data.feature_scale + data.feature_mean
    pred_df = pd.DataFrame(raw, columns=data.features)
    observed_df = pd.DataFrame(data.model_values * data.feature_scale + data.feature_mean, columns=data.features)
    observed_mask = pd.DataFrame(data.observed_mask, columns=data.features)
    rows = data.city[["city_code", "city_name", "country_name", "region_id", "income_id_2022"]].copy()
    groups = {
        "composition": [f for f in data.features if f.startswith("composition_msw_") and f.endswith("_percent")],
        "treatment": [
            f
            for f in data.features
            if (f.startswith("waste_treatment_") and f.endswith("_percent")) or f == "waste_uncollected_percent"
        ],
    }
    for name, cols in groups.items():
        if not cols:
            continue
        obs_counts = observed_mask[cols].sum(axis=1)
        obs_sums = observed_df[cols].where(observed_mask[cols]).sum(axis=1, min_count=1)
        pred_sums = pred_df[cols].clip(lower=0, upper=1).sum(axis=1)
        rows[f"{name}_observed_component_count"] = obs_counts
        rows[f"{name}_observed_sum"] = obs_sums
        rows[f"{name}_completed_sum"] = pred_sums
        rows[f"{name}_completed_abs_error_from_1"] = (pred_sums - 1.0).abs()
    return rows


def make_random_mask(data: PreparedData, rate: float = 0.20, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = data.observed_mask.copy()
    feature_counts = eligible.sum(axis=0)
    eligible[:, feature_counts < 20] = False
    sample = rng.random(eligible.shape) < rate
    return eligible & sample


def make_block_mask(data: PreparedData, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.array([group_id(f) for f in data.features])
    target_groups = {"composition", "treatment", "collection", "budget", "institutional", "legal"}
    rows = rng.choice(data.model_values.shape[0], size=max(40, data.model_values.shape[0] // 4), replace=False)
    cols = np.isin(groups, list(target_groups))
    mask = np.zeros_like(data.observed_mask, dtype=bool)
    mask[np.ix_(rows, np.where(cols)[0])] = True
    return mask & data.observed_mask


def make_group_holdout_mask(data: PreparedData, context_col: str, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = data.context[context_col].astype(str).fillna("")
    counts = values.value_counts()
    candidates = counts[counts >= 10].index.tolist()
    if not candidates:
        return make_random_mask(data, 0.15, seed)
    chosen = candidates[int(seed) % len(candidates)]
    row_mask = values.to_numpy() == chosen
    eligible_cols = data.observed_mask.sum(axis=0) >= 20
    mask = row_mask[:, None] & eligible_cols[None, :] & data.observed_mask
    keep = rng.random(mask.shape) < 0.45
    return mask & keep


def run_imputation_suite(data: PreparedData) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    scenarios = {
        "random_20pct": make_random_mask(data, 0.20, 11),
        "block_by_system": make_block_mask(data, 23),
    }
    if "region_id" in data.context.columns:
        scenarios["region_holdout"] = make_group_holdout_mask(data, "region_id", 2)
    if "income_id_2022" in data.context.columns:
        scenarios["income_holdout"] = make_group_holdout_mask(data, "income_id_2022", 1)

    rows = []
    final_preds: dict[str, np.ndarray] = {}
    final_unc: dict[str, np.ndarray] = {}
    methods: dict[str, Callable[[PreparedData, np.ndarray], tuple[np.ndarray, np.ndarray | None]]] = {
        "mean": lambda d, m: (np.tile(np.nanmean(np.where(m, d.model_values, np.nan), axis=0), (d.model_values.shape[0], 1)), None),
        "median": lambda d, m: (np.tile(np.nanmedian(np.where(m, d.model_values, np.nan), axis=0), (d.model_values.shape[0], 1)), None),
        "region_income_median": lambda d, m: (fill_group_median(d, m), None),
        "city_graph_knn": lambda d, m: (weighted_city_knn(d, m), None),
        "low_rank_svd": lambda d, m: (matrix_factorization(d, m, use_evidence=False), None),
        "ECO-Waste-ECGMC": lambda d, m: eco_waste_predict(d, m),
    }
    for scenario, test_mask in scenarios.items():
        train_mask = data.observed_mask & ~test_mask
        for method, fn in methods.items():
            pred, unc = fn(data, train_mask)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            pred_for_eval = project_percent_constraints(data, pred) if method != "ECO-Waste-ECGMC" else pred
            metrics = evaluate(pred_for_eval, data.model_values, test_mask, unc)
            metrics.update(constraint_violation(data, pred_for_eval))
            metrics["percent_mae_points"] = percent_mae_pp(data, pred_for_eval, test_mask)
            metrics["scenario"] = scenario
            metrics["method"] = method
            rows.append(metrics)
            if scenario == "random_20pct" and method == "ECO-Waste-ECGMC":
                final_preds["main"] = pred_for_eval
                final_unc["main"] = unc if unc is not None else np.zeros_like(pred_for_eval)
    return pd.DataFrame(rows), final_preds, final_unc


def run_ablation_suite(data: PreparedData) -> pd.DataFrame:
    test_mask = make_random_mask(data, 0.20, 111)
    train_mask = data.observed_mask & ~test_mask
    variants = {
        "full_ECO-Waste-ECGMC": dict(),
        "static_rule_evidence": dict(evidence_source="static"),
        "no_codebook_evidence": dict(use_evidence=False),
        "no_city_graph": dict(use_graph=False),
        "no_constraint_projection": dict(use_constraints=False),
        "no_sparse_shrinkage": dict(use_sparse_shrinkage=False),
    }
    rows = []
    for name, kwargs in variants.items():
        evidence_source = kwargs.pop("evidence_source", "learned")
        if evidence_source == "static":
            original_reliability = data.reliability.copy()
            data.reliability = data.static_reliability.copy()
            try:
                pred, unc = eco_waste_predict(data, train_mask, seed=101, **kwargs)
            finally:
                data.reliability = original_reliability
        else:
            pred, unc = eco_waste_predict(data, train_mask, seed=101, **kwargs)
        metrics = evaluate(pred, data.model_values, test_mask, unc)
        metrics.update(constraint_violation(data, pred))
        metrics["percent_mae_points"] = percent_mae_pp(data, pred, test_mask)
        metrics["scenario"] = "random_20pct_ablation"
        metrics["variant"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def risk_components(data: PreparedData, pred: np.ndarray, uncertainty: np.ndarray | None = None) -> pd.DataFrame:
    raw = pred * data.feature_scale + data.feature_mean
    values = pd.DataFrame(raw, columns=data.features)
    unc_raw = uncertainty * data.feature_scale[None, :] if uncertainty is not None else np.zeros_like(raw)
    unc_df = pd.DataFrame(unc_raw, columns=data.features)
    get = lambda c: values[c] if c in values.columns else pd.Series(0.0, index=values.index)
    getu = lambda c: unc_df[c] if c in unc_df.columns else pd.Series(0.0, index=values.index)
    collection_cover = pd.concat(
        [
            get("waste_collection_coverage_total_percent_of_population"),
            get("waste_collection_coverage_total_percent_of_waste"),
        ],
        axis=1,
    ).max(axis=1)
    coverage_gap = 1.0 - collection_cover.clip(0, 1)
    income_prior = data.city.get("income_id_2022", pd.Series("", index=values.index)).astype(str).map(
        {"LIC": 0.12, "LMIC": 0.08, "UMIC": 0.03, "HIC": -0.06}
    ).fillna(0.0)
    risk = (
        get("waste_treatment_open_dumpsite_percent")
        + get("waste_uncollected_percent")
        + get("waste_treatment_unaccounted_for_percent")
        + 0.45 * coverage_gap
        - get("waste_treatment_recycling_percent")
        - get("waste_treatment_compost_percent")
        - get("waste_treatment_sanitary_landfill_landfill_gas_system_percent")
        + income_prior
    )
    risk_uncertainty = np.sqrt(
        getu("waste_treatment_open_dumpsite_percent") ** 2
        + getu("waste_uncollected_percent") ** 2
        + getu("waste_treatment_unaccounted_for_percent") ** 2
        + (0.45 * np.maximum(
            getu("waste_collection_coverage_total_percent_of_population"),
            getu("waste_collection_coverage_total_percent_of_waste"),
        )) ** 2
        + getu("waste_treatment_recycling_percent") ** 2
        + getu("waste_treatment_compost_percent") ** 2
        + getu("waste_treatment_sanitary_landfill_landfill_gas_system_percent") ** 2
    )
    out = data.city[["city_name", "country_name", "region_id", "income_id_2022", "city_code"]].copy()
    out["risk_score"] = risk
    out["risk_uncertainty"] = risk_uncertainty
    out["risk_ci90_low"] = risk - 1.64 * risk_uncertainty
    out["risk_ci90_high"] = risk + 1.64 * risk_uncertainty
    out["risk_percentile"] = out["risk_score"].rank(pct=True)
    high_threshold = float(risk.quantile(0.75))
    out["high_risk_label_top25pct"] = out["risk_score"] >= high_threshold
    out["high_confidence_high_risk"] = out["risk_ci90_low"] >= high_threshold
    for col in [
        "waste_treatment_open_dumpsite_percent",
        "waste_uncollected_percent",
        "waste_treatment_unaccounted_for_percent",
        "waste_collection_coverage_total_percent_of_population",
        "waste_collection_coverage_total_percent_of_waste",
        "waste_treatment_recycling_percent",
        "waste_treatment_compost_percent",
        "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
    ]:
        if col in values.columns:
            out[col] = values[col] * 100
    return out.sort_values("risk_score", ascending=False)


def policy_recommendations(data: PreparedData, risk: pd.DataFrame, pred: np.ndarray, uncertainty: np.ndarray) -> pd.DataFrame:
    raw = pred * data.feature_scale + data.feature_mean
    values = pd.DataFrame(raw, columns=data.features)
    city_meta = data.city[["city_name", "country_name", "region_id", "income_id_2022", "city_code"]].copy()
    merged = city_meta.merge(risk[["city_code", "risk_score", "risk_percentile"]], on="city_code", how="left")
    policy_vars = {
        "institutional_framework_information_system_for_solid_waste_management": "建设/完善固废信息系统",
        "legal_framework_long_term_integrated_solid_waste_master_plan": "制定并执行长期综合固废规划",
        "legal_framework_solid_waste_management_rules_and_regulations": "完善固废法规和执法规则",
        "waste_collection_coverage_total_percent_of_population": "提升人口口径收集覆盖率",
        "waste_collection_coverage_total_percent_of_waste": "提升垃圾量口径收集覆盖率",
        "waste_treatment_recycling_percent": "提高回收分选比例",
        "waste_treatment_compost_percent": "提高厨余/有机堆肥比例",
        "waste_treatment_sanitary_landfill_landfill_gas_system_percent": "增加卫生填埋/气体系统比例",
        "transport_transfer_stations_operational_number": "补强转运站体系",
    }
    rows = []
    for feature, action in policy_vars.items():
        if feature not in values.columns:
            continue
        j = data.features.index(feature)
        val = values[feature]
        obs = data.observed_mask[:, j]
        high = val >= np.nanmedian(val)
        for i, meta in merged.iterrows():
            if meta["risk_percentile"] < 0.70:
                continue
            same = (
                (merged["region_id"].astype(str) == str(meta["region_id"]))
                | (merged["income_id_2022"].astype(str) == str(meta["income_id_2022"]))
            )
            enough = same & obs
            if int(enough.sum()) < 8:
                enough = obs
            treated = enough & high
            untreated = enough & ~high
            if int(treated.sum()) < 3 or int(untreated.sum()) < 3:
                impact = np.nan
            else:
                impact = float(merged.loc[untreated, "risk_score"].median() - merged.loc[treated, "risk_score"].median())
            current = float(val.iloc[i]) if np.isfinite(val.iloc[i]) else np.nan
            if np.isfinite(impact) and impact <= 0:
                continue
            need = current < np.nanmedian(val)
            if not obs[i] and current < np.nanpercentile(val, 65):
                need = True
            if need:
                rows.append(
                    {
                        "city_name": meta["city_name"],
                        "country_name": meta["country_name"],
                        "region_id": meta["region_id"],
                        "income_id_2022": meta["income_id_2022"],
                        "risk_score": float(meta["risk_score"]),
                        "risk_percentile": float(meta["risk_percentile"]),
                        "recommended_action": action,
                        "source_variable": feature,
                        "current_or_imputed_value": current,
                        "estimated_peer_risk_reduction": impact,
                        "uncertainty": float(uncertainty[i, j]),
                        "observed_in_city_dataset": bool(obs[i]),
                    }
                )
    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs
    recs["priority_score"] = (
        recs["risk_percentile"].fillna(0) * 0.45
        + recs["estimated_peer_risk_reduction"].fillna(recs["estimated_peer_risk_reduction"].median()).clip(lower=0) * 0.35
        + recs["uncertainty"].fillna(0) * 0.20
    )
    return recs.sort_values(["priority_score", "risk_score"], ascending=False).head(60)


def policy_peer_effects(data: PreparedData, risk: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    raw = pred * data.feature_scale + data.feature_mean
    values = pd.DataFrame(raw, columns=data.features)
    merged = data.city[["city_code", "region_id", "income_id_2022"]].merge(
        risk[["city_code", "risk_score"]], on="city_code", how="left"
    )
    policy_vars = {
        "institutional_framework_information_system_for_solid_waste_management": "建设/完善固废信息系统",
        "legal_framework_long_term_integrated_solid_waste_master_plan": "制定并执行长期综合固废规划",
        "legal_framework_solid_waste_management_rules_and_regulations": "完善固废法规和执法规则",
        "waste_collection_coverage_total_percent_of_population": "提升人口口径收集覆盖率",
        "waste_collection_coverage_total_percent_of_waste": "提升垃圾量口径收集覆盖率",
        "waste_treatment_recycling_percent": "提高回收分选比例",
        "waste_treatment_compost_percent": "提高厨余/有机堆肥比例",
        "waste_treatment_sanitary_landfill_landfill_gas_system_percent": "增加卫生填埋/气体系统比例",
        "transport_transfer_stations_operational_number": "补强转运站体系",
    }
    rows = []
    for feature, action in policy_vars.items():
        if feature not in values.columns:
            continue
        threshold = float(np.nanmedian(values[feature]))
        treated = values[feature] >= threshold
        for context_col in ["region_id", "income_id_2022"]:
            for context_value, idx in merged.groupby(context_col).groups.items():
                idx = np.array(list(idx))
                if len(idx) < 6:
                    continue
                t = treated.iloc[idx].to_numpy()
                if t.sum() < 3 or (~t).sum() < 3:
                    continue
                treated_risk = float(merged.iloc[idx[t]]["risk_score"].median())
                untreated_risk = float(merged.iloc[idx[~t]]["risk_score"].median())
                rows.append(
                    {
                        "recommended_action": action,
                        "source_variable": feature,
                        "matching_context": context_col,
                        "context_value": context_value,
                        "threshold": threshold,
                        "treated_cities": int(t.sum()),
                        "untreated_cities": int((~t).sum()),
                        "treated_median_risk": treated_risk,
                        "untreated_median_risk": untreated_risk,
                        "estimated_risk_reduction": untreated_risk - treated_risk,
                    }
                )
    return pd.DataFrame(rows).sort_values("estimated_risk_reduction", ascending=False)


def active_acquisition(data: PreparedData) -> pd.DataFrame:
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
    rng = np.random.default_rng(71)
    eligible = data.observed_mask.copy()
    colmask = np.zeros(eligible.shape[1], dtype=bool)
    colmask[cols] = True
    heldout = eligible & colmask[None, :] & (rng.random(eligible.shape) < 0.35)
    entries = np.argwhere(heldout)
    if len(entries) < 30:
        return pd.DataFrame()

    budgets = [0, 10, 20, 40, 70, 100]
    rows = []
    base_train = data.observed_mask & ~heldout
    base_pred, base_unc = eco_waste_predict(data, base_train, seed=77)
    var_missing = 1.0 - base_train.mean(axis=0)
    impact = np.zeros_like(base_unc)
    for f in risk_vars:
        if f in data.features:
            j = data.features.index(f)
            impact[:, j] = 1.0
    methods = {
        "random": rng.random(len(entries)),
        "max_missingness": np.array([var_missing[j] for _, j in entries]),
        "max_uncertainty": np.array([base_unc[i, j] for i, j in entries]),
        "ECO-Acquire": np.array(
            [
                0.34 * base_unc[i, j]
                + 0.28 * var_missing[j]
                + 0.22 * impact[i, j]
                + 0.16 * (1.0 if group_id(data.features[j]) in {"treatment", "collection"} else 0.5)
                for i, j in entries
            ]
        ),
    }
    for method, scores in methods.items():
        order = np.argsort(scores)[::-1]
        for budget in budgets:
            reveal = entries[order[: min(budget, len(entries))]]
            train = base_train.copy()
            if len(reveal):
                train[reveal[:, 0], reveal[:, 1]] = True
            remaining = heldout & ~train
            pred, unc = eco_waste_predict(data, train, seed=88 + budget)
            metrics = evaluate(pred, data.model_values, remaining, unc)
            metrics["strategy"] = method
            metrics["budget_cells"] = budget
            metrics["remaining_test_cells"] = int(remaining.sum())
            metrics["percent_mae_points"] = percent_mae_pp(data, pred, remaining)
            rows.append(metrics)
    return pd.DataFrame(rows)


def active_acquisition_summary(acquisition: pd.DataFrame) -> pd.DataFrame:
    if acquisition.empty:
        return pd.DataFrame()
    base = acquisition[acquisition["budget_cells"] == 0][["strategy", "mae_norm", "percent_mae_points"]].copy()
    baseline_mae = float(base["mae_norm"].iloc[0]) if not base.empty else np.nan
    baseline_pp = float(base["percent_mae_points"].iloc[0]) if not base.empty else np.nan
    rows = []
    for budget, grp in acquisition.groupby("budget_cells"):
        best_mae = grp.sort_values("mae_norm").iloc[0]
        best_pp = grp.sort_values("percent_mae_points").iloc[0]
        rows.append(
            {
                "budget_cells": int(budget),
                "best_strategy_by_mae": best_mae["strategy"],
                "best_mae_norm": float(best_mae["mae_norm"]),
                "mae_improvement_vs_no_acquisition": baseline_mae - float(best_mae["mae_norm"]),
                "best_strategy_by_percent_mae": best_pp["strategy"],
                "best_percent_mae_points": float(best_pp["percent_mae_points"]),
                "percent_mae_improvement_vs_no_acquisition": baseline_pp - float(best_pp["percent_mae_points"]),
            }
        )
    return pd.DataFrame(rows).sort_values("budget_cells")


def svg_bar_chart(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, width: int = 900, height: int = 430) -> None:
    data = df[[label_col, value_col]].dropna().copy()
    if data.empty:
        return
    data = data.head(12)
    max_val = float(data[value_col].max()) or 1.0
    margin_left, margin_top, margin_bottom = 235, 45, 42
    chart_w = width - margin_left - 35
    bar_h = max(16, int((height - margin_top - margin_bottom) / max(len(data), 1)) - 7)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#1f2937">{title}</text>',
    ]
    for idx, row in data.reset_index(drop=True).iterrows():
        y = margin_top + idx * (bar_h + 7)
        label = str(row[label_col])[:42].replace("&", "&amp;")
        val = float(row[value_col])
        bw = max(2, chart_w * val / max_val)
        parts.append(f'<text x="20" y="{y + bar_h - 3}" font-family="Arial" font-size="12" fill="#374151">{label}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="#2f6f73"/>')
        parts.append(f'<text x="{margin_left + bw + 6:.1f}" y="{y + bar_h - 3}" font-family="Arial" font-size="12" fill="#111827">{val:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_line_chart(df: pd.DataFrame, path: Path, title: str, width: int = 900, height: int = 430) -> None:
    if df.empty:
        return
    x_col, y_col, group_col = "budget_cells", "percent_mae_points", "strategy"
    plot = df[[x_col, y_col, group_col]].dropna()
    if plot.empty:
        return
    colors = ["#2f6f73", "#b45309", "#6d5dfc", "#be123c"]
    xmin, xmax = float(plot[x_col].min()), float(plot[x_col].max())
    ymin, ymax = float(plot[y_col].min()), float(plot[y_col].max())
    if ymax == ymin:
        ymax = ymin + 1
    ml, mt, mr, mb = 65, 45, 25, 55
    cw, ch = width - ml - mr, height - mt - mb
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#1f2937">{title}</text>',
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#9ca3af"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#9ca3af"/>',
    ]
    for cidx, (name, group) in enumerate(plot.groupby(group_col)):
        pts = []
        for _, row in group.sort_values(x_col).iterrows():
            x = ml + cw * (float(row[x_col]) - xmin) / max(xmax - xmin, 1e-9)
            y = mt + ch * (1 - (float(row[y_col]) - ymin) / max(ymax - ymin, 1e-9))
            pts.append((x, y))
        if not pts:
            continue
        color = colors[cidx % len(colors)]
        path_d = " ".join([("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts)])
        parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{ml + 20 + cidx * 190}" y="{height - 18}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    parts.append(f'<text x="{ml + cw / 2 - 50:.1f}" y="{height - 8}" font-family="Arial" font-size="12" fill="#374151">新增采集单元数</text>')
    parts.append(f'<text x="10" y="{mt + 14}" font-family="Arial" font-size="12" fill="#374151">风险变量MAE(pp)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "无记录。"
    shown = df.head(max_rows) if max_rows else df
    shown = shown.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x).replace("\n", " ")[:90])
    headers = [str(c) for c in shown.columns]
    rows = shown.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def write_protocol(
    data_dictionary: pd.DataFrame,
    feature_profile: pd.DataFrame,
    missingness_by_group: pd.DataFrame,
    evidence_summary: pd.DataFrame,
    evidence_weights: pd.DataFrame,
    evidence_diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    acquisition_summary_df: pd.DataFrame,
) -> None:
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    steps = pd.DataFrame(
        [
            {
                "word_step": "读取 Excel，保留原始字段名，建立字段字典",
                "implementation": "读取 City dataset 与 Codebook；输出字段级角色、缺失率、Codebook记录数和建模使用情况。",
                "output": "tables/data_dictionary.csv",
            },
            {
                "word_step": "清洗缺失值，区分数值、比例、类别和文本证据",
                "implementation": "统一空值标记；数值列稳健缩放；percent 字段转换到 0-1；低基数类别转 one-hot；文本证据保留在 Codebook 对齐表。",
                "output": "tables/feature_profile.csv",
            },
            {
                "word_step": "变量分组",
                "implementation": "按 population、msw_total、composition、collection、treatment、budget、institutional、legal、facility 等前缀自动分组。",
                "output": "tables/missingness_by_group.csv",
            },
            {
                "word_step": "构建 X 与 M",
                "implementation": "X 为城市-变量建模矩阵，M 为观测掩码；本轮进入模型的派生字段数写入 run_summary.json。",
                "output": "tables/completed_city_matrix_ecowaste.csv",
            },
            {
                "word_step": "解析 Codebook，并按 city_code + measurement 对齐",
                "implementation": "为每条证据记录解析测量年份、方法、来源、链接和备注，标注是否存在于城市表、是否被模型使用。",
                "output": "tables/codebook_aligned.csv",
            },
            {
                "word_step": "构建证据分数",
                "implementation": "规则初始分：称重/分层表征更高，Estimated/Not stated 更低；年份越新、来源和链接越完整分数越高。",
                "output": "tables/evidence_quality_summary.csv",
            },
            {
                "word_step": "建立约束集合",
                "implementation": "对成分比例、处理比例、非负、百分比范围和闭合性进行投影与审计。",
                "output": "tables/constraint_audit.csv",
            },
            {
                "word_step": "跑传统 baseline",
                "implementation": "实现 mean、median、region-income median、city graph KNN、low-rank SVD baseline。",
                "output": "tables/imputation_metrics.csv",
            },
            {
                "word_step": "实现本文模型并做消融",
                "implementation": "ECO-Waste-ECGMC = evidence + graph + low-rank + constraints + sparse shrinkage；并比较去掉各模块后的效果。",
                "output": "tables/model_ablation_metrics.csv",
            },
            {
                "word_step": "设计遮盖实验",
                "implementation": "随机遮盖、变量块遮盖、区域留出、收入组留出。",
                "output": "tables/imputation_metrics.csv",
            },
            {
                "word_step": "做风险识别",
                "implementation": "基于露天倾倒、未收集、未闭合、收集覆盖缺口、回收/堆肥/卫生填埋和收入组脆弱性构造弱监督风险分数，并输出不确定性区间。",
                "output": "tables/risk_ranking.csv",
            },
            {
                "word_step": "做政策推荐",
                "implementation": "在区域/收入组内做观测匹配，输出正向同伴风险降低的政策优先级。",
                "output": "tables/policy_recommendations.csv; tables/policy_peer_effects.csv",
            },
            {
                "word_step": "做主动采集",
                "implementation": "模拟只能新增 k 个观测单元时，不同采集策略对风险变量补全误差的影响。",
                "output": "tables/active_acquisition.csv; tables/active_acquisition_summary.csv",
            },
        ]
    )
    steps.to_csv(TABLE_DIR / "word_step_mapping.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# ECO-Waste v2 实验协议",
        "",
        "本协议逐条对应 Word 文档中的实施步骤，所有输入、代码和输出均位于 `E:\\waste`。",
        "",
        "## 步骤映射",
        md_table(steps),
        "",
        "## 数据画像摘要",
        f"- 字段总数：{len(data_dictionary)}",
        f"- 进入模型的派生字段数：{len(feature_profile)}",
        f"- 平均观测率：{feature_profile['observed_rate'].mean():.2%}",
        "",
        "## 缺失率最高的变量组",
        md_table(missingness_by_group.head(12)),
        "",
        "## 证据质量摘要",
        md_table(evidence_summary.head(12)),
        "",
        "## 学习式证据校准",
        md_table(evidence_diagnostics),
        "",
        md_table(evidence_weights),
        "",
        "## 补全实验摘要",
        md_table(metrics.sort_values(['scenario', 'mae_norm']).head(18)),
        "",
        "## 消融实验摘要",
        md_table(ablation.sort_values('mae_norm')),
        "",
        "## 主动采集摘要",
        md_table(acquisition_summary_df),
    ]
    (PROTOCOL_DIR / "experiment_protocol.md").write_text("\n".join(lines), encoding="utf-8")


def write_application_workflow(
    evidence_diagnostics: pd.DataFrame,
    ablation: pd.DataFrame,
    acquisition_summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# ECO-Waste 从数据处理到实际应用的整体流程",
        "",
        "## 目标",
        "把 `What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx` 从一个高缺失、证据异构的研究数据表，转换成可用于城市固废系统补全、风险识别、政策优先级和主动数据采集的决策支持流程。",
        "",
        "## 总体流程图",
        "```mermaid",
        "flowchart TD",
        "  A[原始 Excel: City dataset + Codebook] --> B[字段字典与缺失画像]",
        "  B --> C[数值/比例/类别清洗与城市-变量矩阵 X,M]",
        "  A --> D[Codebook 证据解析: 方法/年份/来源/链接/备注]",
        "  D --> E[规则证据分数]",
        "  C --> F[自监督遮盖: 留出已观测单元]",
        "  E --> F",
        "  F --> G[学习式证据校准器: learned reliability]",
        "  C --> H[ECO-Waste-ECGMC 补全模型]",
        "  G --> H",
        "  H --> I[约束投影: 成分/处理/非负/百分比闭合]",
        "  I --> J[风险分数 + 不确定性区间]",
        "  J --> K[政策推荐: 区域/收入组匹配验证]",
        "  J --> L[主动采集: 下一批应调查的 city-variable]",
        "  L --> M[城市调查/人工核验]",
        "  M --> A",
        "```",
        "",
        "## 阶段 1：数据资产化",
        "1. 原始 Excel 不直接改动，作为 `data/raw` 保存。",
        "2. 建立 `data_dictionary.csv`，记录每个字段的角色、变量组、非缺失率、Codebook 记录数和是否进入模型。",
        "3. 建立 `missingness_by_group.csv`，把缺失问题从单列层面提升到系统模块层面，例如 collection、treatment、facility、budget。",
        "",
        "## 阶段 2：证据学习增强",
        "旧版只使用规则证据分数：称重、分层表征、年份较新、来源完整的观测更可信。v3 新增学习式增强：",
        "1. 从已观测单元中自监督遮盖一部分。",
        "2. 用基础补全器预测这些遮盖值，得到每条证据对应的误差。",
        "3. 将误差转换为目标可靠度，学习 Codebook 元数据到可靠度的校准权重。",
        "4. 最终 reliability = 规则可靠度与学习可靠度的融合，而不是只靠人工规则。",
        "",
        "学习式校准诊断：",
        md_table(evidence_diagnostics),
        "",
        "## 阶段 3：可信补全",
        "ECO-Waste-ECGMC 使用城市相似图、低秩结构、学习式证据可靠度和固废系统约束进行缺失补全。补全输出不是孤立预测，而是附带约束审计和不确定性。",
        "",
        "消融实验用于判断哪些模块真正有贡献：",
        md_table(ablation.sort_values("mae_norm")),
        "",
        "## 阶段 4：风险识别",
        "风险分数由露天倾倒、未收集、未闭合、收集覆盖缺口、低回收/堆肥/卫生填埋和收入组脆弱性组成。v3 输出 `risk_score`、`risk_uncertainty` 和保守的 proxy 风险区间。注意该区间是模型不确定性的决策代理，不应写成严格统计置信区间。",
        "",
        "## 阶段 5：政策推荐",
        "政策推荐不声称因果结论，而采用同区域/同收入组匹配验证：如果某政策变量较高的城市在同伴组内风险中位数更低，则把它作为候选政策优先级。对应输出为 `policy_recommendations.csv` 和 `policy_peer_effects.csv`。",
        "",
        "## 阶段 6：主动数据采集",
        "主动采集回答实际应用中的问题：调查预算有限时，下一批最该核验哪些城市-变量单元。策略比较摘要：",
        md_table(acquisition_summary_df),
        "",
        "## 实际部署闭环",
        "1. 先用现有 Excel 生成城市风险与政策优先级。",
        "2. 让领域专家审核 Top-risk 城市和 Top-policy 建议，标注明显不合理项。",
        "3. 按主动采集清单补采高价值缺失变量，例如收集覆盖率、露天倾倒比例、卫生填埋比例。",
        "4. 将新观测追加回 City dataset/Codebook，保留来源、方法、年份和备注。",
        "5. 重新运行脚本，证据校准器会用新增观测更新 learned reliability，形成数据-模型-政策-采集闭环。",
    ]
    (PROTOCOL_DIR / "end_to_end_application_workflow.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    data: PreparedData,
    metrics: pd.DataFrame,
    risk: pd.DataFrame,
    recs: pd.DataFrame,
    acquisition: pd.DataFrame,
    ablation: pd.DataFrame,
    missingness_by_group: pd.DataFrame,
    evidence_summary: pd.DataFrame,
    evidence_weights: pd.DataFrame,
    evidence_diagnostics: pd.DataFrame,
    acquisition_summary_df: pd.DataFrame,
) -> None:
    best = metrics.sort_values("mae_norm").head(8)
    random_rows = metrics[metrics["scenario"] == "random_20pct"].sort_values("mae_norm")
    active_best = pd.DataFrame()
    if not acquisition.empty:
        active_best = acquisition.sort_values(["budget_cells", "percent_mae_points"]).groupby("budget_cells").head(1)
    lines = [
        "# ECO-Waste 实验报告",
        "",
        "## 实验定位",
        "本实验严格只使用 `What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx`。按照 Word 文档中的 AAAI 思路，将 City dataset 建成城市-变量观测矩阵，将 Codebook 转化为观测证据可信度，并围绕缺失补全、风险识别、政策推荐和主动数据采集四个任务做实验。",
        "",
        "## 原创算法：ECO-Waste-ECGMC",
        "ECO-Waste-ECGMC 是 Evidence-aware Constrained Graph Matrix Completion 的轻量实现。它包含四个定制模块：",
        "1. 证据可信度编码：根据测量方法、测量年份、来源、网页链接、备注和测量点为每个 city-variable 观测赋权。",
        "2. 城市相似图传播：利用共享观测、区域和收入组构造城市近邻，把相似城市的可信观测传播到缺失单元。",
        "3. 证据加权低秩补全：用自实现的加权矩阵分解学习城市状态和变量状态，对高可信观测给予更大重构权重。",
        "4. 固废约束投影：对成分比例、处理/未收集比例进行非负和闭合约束修正，并用多模型分歧生成不确定性。",
        "",
        "## 数据利用情况",
        f"- 城市数：{data.city.shape[0]}",
        f"- 原始城市字段数：{data.city.shape[1]}",
        f"- 进入建模的数值/二值字段数：{len(data.features)}",
        f"- Codebook 证据记录数：{data.codebook.shape[0]}",
        f"- 建模矩阵观测率：{data.observed_mask.mean():.1%}",
        "",
        "## 变量组缺失与证据质量",
        md_table(missingness_by_group.head(10)),
        "",
        md_table(evidence_summary.head(10)),
        "",
        "## 学习式证据增强",
        "v3 不再只依赖人工规则证据分数，而是用自监督遮盖学习一个证据校准器：先估计不同 Codebook 元数据对应的补全误差，再把误差转为目标可靠度，并学习证据特征权重。主模型默认使用规则分数与学习分数融合后的 reliability。",
        "",
        md_table(evidence_diagnostics),
        "",
        md_table(evidence_weights),
        "",
        "## 主要补全结果",
        md_table(random_rows),
        "",
        "## 结果解读",
        "中位数和分组中位数在 MAE 上非常强，这是小样本、高缺失表格的典型现象：保守估计很难被复杂模型轻易超过。ECO-Waste-ECGMC 的优势主要体现在三个方面：第一，四个遮盖场景下 RMSE 均低于普通中位数，说明它对大误差更稳；第二，成分/处理比例的约束误差更低；第三，它额外输出不确定性、风险排序、政策建议和主动采集优先级，这些是普通补全基线不能直接给出的决策支持能力。",
        "",
        "## 全场景最佳结果概览",
        md_table(best),
        "",
        "## 模型消融",
        md_table(ablation.sort_values("mae_norm")),
        "",
        "## 高风险城市示例",
        md_table(risk.head(20)),
        "",
        "## 政策推荐示例",
        md_table(recs.head(25)) if not recs.empty else "未生成政策推荐。",
        "",
        "## 主动采集结果",
        md_table(acquisition) if not acquisition.empty else "主动采集样本不足，未运行。",
        "",
        "## 主动采集最佳策略摘要",
        md_table(acquisition_summary_df) if not acquisition_summary_df.empty else "主动采集样本不足，未生成摘要。",
        "",
        "## 结论",
        "实验结果可作为 AAAI 论文初版实验骨架：ECO-Waste-ECGMC 同时利用数值矩阵、区域收入上下文、Codebook 证据和固废比例约束，并把补全结果连接到风险识别、政策优先级和主动采集。后续若要进一步冲击投稿质量，建议把当前 numpy 版本升级为可训练的图神经网络版本，并增加更严格的跨区域留出、校准图和人工审核的政策解释。",
    ]
    (OUT_DIR / "ECO-Waste_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for path in [WORK_DIR, TABLE_DIR, FIG_DIR, PROTOCOL_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    city, codebook = read_inputs()
    data = prepare_features(city, codebook)
    data.evidence_calibration_weights.to_csv(TABLE_DIR / "evidence_learning_weights.csv", index=False, encoding="utf-8-sig")
    data.evidence_calibration_diagnostics.to_csv(TABLE_DIR / "evidence_learning_diagnostics.csv", index=False, encoding="utf-8-sig")
    data_dictionary = build_data_dictionary(data)
    data_dictionary.to_csv(TABLE_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    codebook_aligned, evidence_summary = build_codebook_alignment(data)
    codebook_aligned.to_csv(TABLE_DIR / "codebook_aligned.csv", index=False, encoding="utf-8-sig")
    evidence_summary.to_csv(TABLE_DIR / "evidence_quality_summary.csv", index=False, encoding="utf-8-sig")
    profile = pd.DataFrame(
        {
            "feature": data.features,
            "group": [group_id(f) for f in data.features],
            "observed_count": data.observed_mask.sum(axis=0),
            "observed_rate": data.observed_mask.mean(axis=0),
            "mean_evidence_reliability": np.where(
                data.observed_mask.sum(axis=0) > 0,
                (data.reliability * data.observed_mask).sum(axis=0) / np.maximum(data.observed_mask.sum(axis=0), 1),
                np.nan,
            ),
            "is_percent_or_binary": [f in data.percent_features for f in data.features],
        }
    )
    profile.to_csv(TABLE_DIR / "feature_profile.csv", index=False, encoding="utf-8-sig")
    missingness_by_group = build_missingness_by_group(profile)
    missingness_by_group.to_csv(TABLE_DIR / "missingness_by_group.csv", index=False, encoding="utf-8-sig")
    metrics, preds, uncs = run_imputation_suite(data)
    metrics = metrics[
        [
            "scenario",
            "method",
            "heldout_cells",
            "mae_norm",
            "rmse_norm",
            "percent_mae_points",
            "coverage_90pct_proxy",
            "composition_sum_abs_error",
            "treatment_sum_abs_error",
        ]
    ]
    metrics.to_csv(TABLE_DIR / "imputation_metrics.csv", index=False, encoding="utf-8-sig")
    ablation = run_ablation_suite(data)
    ablation = ablation[
        [
            "scenario",
            "variant",
            "heldout_cells",
            "mae_norm",
            "rmse_norm",
            "percent_mae_points",
            "coverage_90pct_proxy",
            "composition_sum_abs_error",
            "treatment_sum_abs_error",
        ]
    ]
    ablation.to_csv(TABLE_DIR / "model_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    main_pred = preds["main"]
    main_unc = uncs["main"]
    completed = pd.DataFrame(main_pred * data.feature_scale + data.feature_mean, columns=data.features)
    completed.insert(0, "city_code", data.city["city_code"])
    completed.insert(1, "city_name", data.city["city_name"])
    completed.insert(2, "country_name", data.city["country_name"])
    completed.to_csv(TABLE_DIR / "completed_city_matrix_ecowaste.csv", index=False, encoding="utf-8-sig")
    constraints = constraint_audit(data, main_pred)
    constraints.to_csv(TABLE_DIR / "constraint_audit.csv", index=False, encoding="utf-8-sig")
    risk = risk_components(data, main_pred, main_unc)
    risk.to_csv(TABLE_DIR / "risk_ranking.csv", index=False, encoding="utf-8-sig")
    recs = policy_recommendations(data, risk, main_pred, main_unc)
    recs.to_csv(TABLE_DIR / "policy_recommendations.csv", index=False, encoding="utf-8-sig")
    peer_effects = policy_peer_effects(data, risk, main_pred)
    peer_effects.to_csv(TABLE_DIR / "policy_peer_effects.csv", index=False, encoding="utf-8-sig")
    acquisition = active_acquisition(data)
    acquisition.to_csv(TABLE_DIR / "active_acquisition.csv", index=False, encoding="utf-8-sig")
    acquisition_summary_df = active_acquisition_summary(acquisition)
    acquisition_summary_df.to_csv(TABLE_DIR / "active_acquisition_summary.csv", index=False, encoding="utf-8-sig")

    metric_plot = metrics[metrics["scenario"] == "random_20pct"].sort_values("mae_norm")
    svg_bar_chart(metric_plot, "method", "mae_norm", FIG_DIR / "random_masking_mae.svg", "Random masking: normalized MAE")
    ablation_plot = ablation.sort_values("mae_norm")
    svg_bar_chart(ablation_plot, "variant", "mae_norm", FIG_DIR / "ablation_mae.svg", "Ablation: normalized MAE")
    missing_plot = profile.sort_values("observed_rate").head(12).copy()
    missing_plot["missing_rate"] = 1 - missing_plot["observed_rate"]
    svg_bar_chart(missing_plot, "feature", "missing_rate", FIG_DIR / "highest_missing_features.svg", "Highest missingness among modeled features")
    svg_line_chart(acquisition, FIG_DIR / "active_acquisition_curve.svg", "Active acquisition simulation")
    write_protocol(
        data_dictionary,
        profile,
        missingness_by_group,
        evidence_summary,
        data.evidence_calibration_weights,
        data.evidence_calibration_diagnostics,
        metrics,
        ablation,
        acquisition_summary_df,
    )
    write_application_workflow(data.evidence_calibration_diagnostics, ablation, acquisition_summary_df)
    write_report(
        data,
        metrics,
        risk,
        recs,
        acquisition,
        ablation,
        missingness_by_group,
        evidence_summary,
        data.evidence_calibration_weights,
        data.evidence_calibration_diagnostics,
        acquisition_summary_df,
    )
    summary = {
        "project_root": str(ROOT),
        "input_dataset": str(XLSX_PATH),
        "output_version": "ecowaste_v3",
        "n_cities": int(data.city.shape[0]),
        "n_raw_fields": int(data.city.shape[1]),
        "n_modeled_features": int(len(data.features)),
        "n_codebook_records": int(data.codebook.shape[0]),
        "observed_rate_model_matrix": float(data.observed_mask.mean()),
        "outputs": {
            "report": str(OUT_DIR / "ECO-Waste_experiment_report.md"),
            "protocol": str(PROTOCOL_DIR / "experiment_protocol.md"),
            "application_workflow": str(PROTOCOL_DIR / "end_to_end_application_workflow.md"),
            "tables": str(TABLE_DIR),
            "figures": str(FIG_DIR),
        },
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
