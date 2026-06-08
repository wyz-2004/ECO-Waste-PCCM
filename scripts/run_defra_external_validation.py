from __future__ import annotations

import argparse
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


RAW_DIR = ROOT / "data" / "raw" / "defra_wastedataflow"
CLEAN_DIR = ROOT / "data" / "processed" / "defra_wastedataflow"
OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
PROTOCOL_DIR = OUT_DIR / "protocol"

SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
STRUCTURED_SEEDS = SEEDS[:5]
ACQUISITION_SEEDS = SEEDS[:3]
EXPERT_METHODS = ["Median", "KNN", "SoftImpute", "Low-rank SVD"]
METHODS = EXPERT_METHODS + ["Ours-L1", "Ours-L2"]
EPS = 1e-9

CHANNEL_MAP = {
    "Q010": "kerbside_household_recycling",
    "Q011": "commercial_nonhousehold_recycling",
    "Q012": "voluntary_kerbside_recycling",
    "Q016": "civic_amenity_recycling",
    "Q017": "la_bring_site_recycling",
    "Q018": "other_recycling_schemes",
    "Q023": "residual_disposal_collection",
    "Q033": "voluntary_bring_site_reuse",
    "Q034": "street_bin_recycling",
}


def ensure_dirs() -> None:
    for path in [CLEAN_DIR, OUT_DIR, TABLE_DIR, PROTOCOL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def slug(value: object, limit: int = 44) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = "blank"
    return text[:limit].strip("_")


def financial_year_from_name(path: Path) -> str:
    match = re.search(r"(20\d{2})[-_](\d{2})", path.stem)
    if not match:
        raise ValueError(f"Could not infer financial year from {path.name}")
    start = int(match.group(1))
    end = int(str(start)[:2] + match.group(2))
    return f"{start}_{end}"


def authority_type(authority: str) -> str:
    text = authority.lower()
    if "london borough" in text:
        return "london_borough"
    if "metropolitan borough" in text:
        return "metropolitan_borough"
    if "county council" in text:
        return "county_council"
    if "district council" in text:
        return "district_council"
    if "city council" in text:
        return "city_council"
    if "borough council" in text:
        return "borough_council"
    if "council of the isles" in text:
        return "unitary_or_special"
    return "unitary_or_other"


def clean_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace(
        {
            "": np.nan,
            "-": np.nan,
            "--": np.nan,
            "*": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "N/A": np.nan,
            "n/a": np.nan,
            "Not available": np.nan,
            "not available": np.nan,
            "Suppressed": np.nan,
            "suppressed": np.nan,
        }
    )
    text = text.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def read_raw() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No Defra CSV files found under {RAW_DIR}")
    for path in csvs:
        frame = pd.read_csv(path, header=1, encoding="utf-8-sig", dtype=str)
        frame["source_file"] = path.name
        frame["financial_year"] = financial_year_from_name(path)
        frame["financial_year_start"] = int(frame["financial_year"].str[:4].iloc[0])
        frame["data_numeric"] = clean_numeric(frame["Data"])
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["Authority"] = raw["Authority"].astype(str).str.strip()
    raw = raw[raw["Authority"].ne("") & raw["QuestionNumber"].notna()].copy()
    return raw


def aggregation_kind(label: str) -> str:
    lower = label.lower()
    if any(key in lower for key in ["tonnage", "tonnes", "incident", "vehicles", "fridges"]):
        return "sum"
    return "mean"


def direct_kind(label: str) -> str:
    lower = label.lower()
    if "percentage" in lower or "% " in lower or " % " in lower or lower.endswith("%"):
        return "rate"
    if "tonnage" in lower or "tonnes" in lower:
        return "tonnage"
    if "population" in lower or "area" in lower or "density" in lower or "household" in lower or "dwelling" in lower:
        return "profile"
    if "frequency" in lower:
        return "service"
    if "incident" in lower or "vehicle" in lower or "fridge" in lower or "site" in lower:
        return "count"
    return "numeric"


def make_direct_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = raw[raw["data_numeric"].notna()].copy()
    direct = direct[direct["CollateText"].isna() | direct["CollateText"].astype(str).str.strip().eq("")]
    direct = direct[~direct["QuestionNumber"].isin(["Q014"])].copy()
    label = (
        direct["QuestionNumber"].fillna("")
        + " "
        + direct["RowText"].fillna("")
        + " "
        + direct["ColText"].fillna("")
        + " "
        + direct["MaterialGroup"].fillna("")
    )
    direct["kind"] = label.map(direct_kind)
    direct["aggregation"] = label.map(aggregation_kind)
    direct["feature"] = [
        f"{kind}__{str(q).lower()}_r{str(row_id)}_c{str(col_id)}_{slug(row)}_{slug(col, 28)}"
        for kind, q, row_id, col_id, row, col in zip(
            direct["kind"],
            direct["QuestionNumber"],
            direct["RowIdent"].fillna("x"),
            direct["ColIdent"].fillna("x"),
            direct["RowText"],
            direct["ColText"],
        )
    ]
    keys = ["Authority", "financial_year", "feature"]
    summed = (
        direct[direct["aggregation"].eq("sum")]
        .groupby(keys, dropna=False)["data_numeric"]
        .sum(min_count=1)
    )
    averaged = (
        direct[direct["aggregation"].ne("sum")]
        .groupby(keys, dropna=False)["data_numeric"]
        .mean()
    )
    long = pd.concat([summed, averaged]).reset_index()
    wide = long.pivot_table(index=["Authority", "financial_year"], columns="feature", values="data_numeric", aggfunc="mean")

    dictionary = (
        direct[
            [
                "feature",
                "kind",
                "aggregation",
                "QuestionNumber",
                "QuText",
                "RowText",
                "ColText",
                "MaterialGroup",
                "RowIdent",
                "ColIdent",
            ]
        ]
        .drop_duplicates("feature")
        .sort_values("feature")
    )
    return wide, dictionary


def is_tonnage_record(frame: pd.DataFrame) -> pd.Series:
    text = (frame["RowText"].fillna("") + " " + frame["ColText"].fillna("")).str.lower()
    has_tonnage = text.str.contains("tonnage|tonnes", regex=True)
    excluded = text.str.contains("rejected|disposed|number of|no\\. of|households|sites|frequency|percentage|weighed", regex=True)
    return has_tonnage & ~excluded & frame["data_numeric"].notna()


def make_share_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tonnage = raw[raw["QuestionNumber"].isin(CHANNEL_MAP) & is_tonnage_record(raw)].copy()
    tonnage["channel"] = tonnage["QuestionNumber"].map(CHANNEL_MAP)
    tonnage["material_clean"] = tonnage["MaterialGroup"].map(lambda x: slug(x, 32))
    tonnage.loc[tonnage["MaterialGroup"].isna() | tonnage["material_clean"].eq("blank"), "material_clean"] = "other_materials"

    channel_amount = (
        tonnage.groupby(["Authority", "financial_year", "channel"], dropna=False)["data_numeric"]
        .sum(min_count=1)
        .reset_index()
    )
    channel_wide = channel_amount.pivot_table(
        index=["Authority", "financial_year"], columns="channel", values="data_numeric", aggfunc="sum"
    )
    channel_wide = channel_wide.reindex(columns=list(CHANNEL_MAP.values())).fillna(0.0)
    channel_total = channel_wide.sum(axis=1)
    channel_share = channel_wide.div(channel_total.replace(0, np.nan), axis=0)
    channel_share.columns = [f"waste_treatment_{c}_percent" for c in channel_share.columns]
    channel_amount_features = channel_wide.copy()
    channel_amount_features.columns = [f"tonnage__channel_{c}" for c in channel_amount_features.columns]

    material_amount = (
        tonnage.groupby(["Authority", "financial_year", "material_clean"], dropna=False)["data_numeric"]
        .sum(min_count=1)
        .reset_index()
    )
    totals = material_amount.groupby("material_clean")["data_numeric"].sum().sort_values(ascending=False)
    keep_materials = [m for m in totals.index.tolist() if m != "other_materials"][:9]
    material_amount["material_bucket"] = np.where(
        material_amount["material_clean"].isin(keep_materials),
        material_amount["material_clean"],
        "other_materials",
    )
    material_wide = material_amount.pivot_table(
        index=["Authority", "financial_year"], columns="material_bucket", values="data_numeric", aggfunc="sum"
    )
    material_cols = keep_materials + (["other_materials"] if "other_materials" in material_amount["material_bucket"].unique() else [])
    material_wide = material_wide.reindex(columns=material_cols).fillna(0.0)
    material_total = material_wide.sum(axis=1)
    material_share = material_wide.div(material_total.replace(0, np.nan), axis=0)
    material_share.columns = [f"composition_msw_{c}_percent" for c in material_share.columns]
    material_amount_features = material_wide.copy()
    material_amount_features.columns = [f"tonnage__material_{c}" for c in material_amount_features.columns]

    wide = pd.concat([channel_amount_features, channel_share, material_amount_features, material_share], axis=1)
    dictionary_rows: list[dict[str, object]] = []
    for col in channel_amount_features.columns:
        dictionary_rows.append({"feature": col, "kind": "tonnage", "aggregation": "sum", "QuestionNumber": "multi", "QuText": "Annual channel tonnage from Defra WasteDataFlow", "RowText": "", "ColText": "", "MaterialGroup": ""})
    for col in channel_share.columns:
        dictionary_rows.append({"feature": col, "kind": "rate", "aggregation": "derived_share", "QuestionNumber": "multi", "QuText": "Annual collection-channel closure share from Defra WasteDataFlow", "RowText": "", "ColText": "", "MaterialGroup": ""})
    for col in material_amount_features.columns:
        dictionary_rows.append({"feature": col, "kind": "tonnage", "aggregation": "sum", "QuestionNumber": "multi", "QuText": "Annual material-group tonnage from Defra WasteDataFlow", "RowText": "", "ColText": "", "MaterialGroup": col.rsplit("_", 1)[-1]})
    for col in material_share.columns:
        dictionary_rows.append({"feature": col, "kind": "rate", "aggregation": "derived_share", "QuestionNumber": "multi", "QuText": "Annual material composition closure share from Defra WasteDataFlow", "RowText": "", "ColText": "", "MaterialGroup": col.rsplit("_", 2)[-2]})
    return wide, pd.DataFrame(dictionary_rows)


def select_direct_columns(direct_wide: pd.DataFrame, max_columns: int = 48) -> list[str]:
    observed = direct_wide.notna().sum(axis=0)
    variance = direct_wide.var(axis=0, skipna=True)
    minimum = max(120, int(0.18 * len(direct_wide)))
    eligible = observed[(observed >= minimum) & (variance > EPS)].index.tolist()
    ranked = sorted(eligible, key=lambda c: (observed[c], variance[c]), reverse=True)
    return ranked[:max_columns]


def build_wide_workbook(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        raw[["Authority", "financial_year"]]
        .drop_duplicates()
        .sort_values(["Authority", "financial_year"])
        .set_index(["Authority", "financial_year"])
    )
    direct_wide, direct_dict = make_direct_features(raw)
    share_wide, share_dict = make_share_features(raw)
    direct_keep = select_direct_columns(direct_wide)
    wide = index.join(direct_wide[direct_keep], how="left").join(share_wide, how="left")
    wide = wide.loc[:, wide.notna().sum(axis=0) > 0]
    nonconstant = wide.var(axis=0, skipna=True) > EPS
    wide = wide.loc[:, nonconstant]
    dictionary = pd.concat([direct_dict[direct_dict["feature"].isin(direct_keep)], share_dict], ignore_index=True)
    dictionary = dictionary[dictionary["feature"].isin(wide.columns)].drop_duplicates("feature").sort_values("feature")

    meta = wide.reset_index()[["Authority", "financial_year"]].copy()
    meta["financial_year_start"] = meta["financial_year"].str[:4].astype(int)
    meta["authority_type"] = meta["Authority"].map(authority_type)
    meta["city_code"] = [f"defra_{slug(a, 38)}_{fy}" for a, fy in zip(meta["Authority"], meta["financial_year"])]
    meta["city_name"] = meta["Authority"]
    meta["country_name"] = "England"
    meta["region_id"] = meta["authority_type"]
    meta["income_id_2022"] = meta["financial_year"]
    return wide.reset_index(drop=True), meta, dictionary


def prepared_from_wide(wide: pd.DataFrame, meta: pd.DataFrame, dictionary: pd.DataFrame) -> base.PreparedData:
    raw_values = wide.copy()
    features = raw_values.columns.tolist()
    percent_features = {
        c
        for c in features
        if c.startswith("composition_msw_") or c.startswith("waste_treatment_") or c.startswith("rate__")
    }
    binary_features: set[str] = set()
    model_df = raw_values.copy()
    for col in features:
        values = pd.to_numeric(model_df[col], errors="coerce")
        if col in percent_features:
            if values.dropna().size and values.dropna().quantile(0.95) > 1.5:
                values = values / 100.0
            model_df[col] = values.clip(0, 1)
        else:
            if values.dropna().size and values.dropna().quantile(0.95) > 1000 and values.min(skipna=True) >= 0:
                values = np.log1p(values)
            model_df[col] = values

    observed_mask = model_df.notna().to_numpy()
    model_values_raw = model_df.to_numpy(dtype=float)
    feature_mean = np.nanmedian(model_values_raw, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    q75 = np.nanpercentile(model_values_raw, 75, axis=0)
    q25 = np.nanpercentile(model_values_raw, 25, axis=0)
    feature_scale = q75 - q25
    std = np.nanstd(model_values_raw, axis=0)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > EPS), feature_scale, std)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > EPS), feature_scale, 1.0)
    model_values = (model_values_raw - feature_mean) / feature_scale
    model_values = np.where(np.isfinite(model_values), np.clip(model_values, -8, 8), np.nan)

    n, p = model_values.shape
    reliability = np.full((n, p), 0.80, dtype=float)
    reliability = np.where(observed_mask, reliability, 0.60)
    evidence_features = np.zeros((n, p, len(base.EVIDENCE_FEATURE_NAMES)), dtype=float)
    evidence_features[:, :, :] = base.evidence_feature_vector(None)[None, None, :]
    evidence_features[:, :, 1] = 0.80
    evidence_features[:, :, 7] = 1.0
    year_recency = np.clip((meta["financial_year_start"].to_numpy(dtype=float) - 2000.0) / 25.0, 0.05, 1.0)
    evidence_features[:, :, 8] = year_recency[:, None]
    evidence_features[:, :, 9] = 1.0
    evidence_features[:, :, 10] = 1.0

    city = meta.copy()
    context = city[["region_id", "income_id_2022"]].copy()
    return base.PreparedData(
        city=city,
        codebook=dictionary,
        features=features,
        percent_features=percent_features,
        binary_features=binary_features,
        raw_values=raw_values,
        model_values=model_values,
        observed_mask=observed_mask,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        static_reliability=reliability.copy(),
        reliability=reliability,
        evidence_features=evidence_features,
        evidence_feature_names=base.EVIDENCE_FEATURE_NAMES.copy(),
        evidence_calibration_weights=pd.DataFrame(),
        evidence_calibration_diagnostics=pd.DataFrame(),
        context=context,
    )


def write_dataset_artifacts(raw: pd.DataFrame, wide: pd.DataFrame, meta: pd.DataFrame, dictionary: pd.DataFrame, data: base.PreparedData) -> pd.DataFrame:
    modeled = pd.concat([meta[["city_code", "Authority", "financial_year", "authority_type"]], wide], axis=1)
    modeled.to_csv(CLEAN_DIR / "defra_modeled_authority_year_workbook.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(CLEAN_DIR / "defra_feature_dictionary.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "source": "Defra/WasteDataFlow England annual collection CSVs",
                "source_csv_files": int(raw["source_file"].nunique()),
                "raw_long_records": int(len(raw)),
                "authority_year_rows": int(len(wide)),
                "local_authorities": int(meta["Authority"].nunique()),
                "financial_years": ", ".join(sorted(meta["financial_year"].unique())),
                "modeled_features": int(len(data.features)),
                "direct_features": int(sum("__" in f for f in data.features)),
                "closure_share_features": int(sum(f.startswith("composition_msw_") or f.startswith("waste_treatment_") for f in data.features)),
                "percent_features": int(len(data.percent_features)),
                "observed_cell_rate": float(data.observed_mask.mean()),
                "missing_cell_rate": float(1.0 - data.observed_mask.mean()),
                "composition_features": int(sum(f.startswith("composition_msw_") for f in data.features)),
                "treatment_features": int(sum(f.startswith("waste_treatment_") for f in data.features)),
            }
        ]
    )
    summary.to_csv(TABLE_DIR / "defra_dataset_summary.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(TABLE_DIR / "defra_feature_dictionary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_defra_data() -> tuple[base.PreparedData, pd.DataFrame]:
    raw = read_raw()
    wide, meta, dictionary = build_wide_workbook(raw)
    data = prepared_from_wide(wide, meta, dictionary)
    summary = write_dataset_artifacts(raw, wide, meta, dictionary, data)
    return data, summary


def random_mask(data: base.PreparedData, seed: int) -> np.ndarray:
    return base.make_random_mask(data, 0.20, seed)


def latest_year_mask(data: base.PreparedData, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 501)
    latest = sorted(data.city["financial_year"].unique())[-1]
    rows = data.city["financial_year"].eq(latest).to_numpy()
    return data.observed_mask & rows[:, None] & (rng.random(data.observed_mask.shape) < 0.35)


def block_mask(data: base.PreparedData, seed: int, family: str) -> np.ndarray:
    rng = np.random.default_rng(seed + (701 if family == "treatment" else 901))
    if family == "treatment":
        cols = np.array([f.startswith("waste_treatment_") or f.startswith("tonnage__channel_") for f in data.features])
    elif family == "composition":
        cols = np.array([f.startswith("composition_msw_") or f.startswith("tonnage__material_") for f in data.features])
    else:
        raise ValueError(family)
    chosen_rows = rng.random(data.model_values.shape[0]) < 0.38
    return data.observed_mask & chosen_rows[:, None] & cols[None, :]


def protocol_masks(data: base.PreparedData, seed: int) -> dict[str, np.ndarray]:
    return {
        "random_20pct": random_mask(data, seed),
        "latest_year_2024_25": latest_year_mask(data, seed),
        "treatment_block": block_mask(data, seed, "treatment"),
        "composition_block": block_mask(data, seed, "composition"),
    }


def generic_uncertainty(data: base.PreparedData, train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    observed = np.where(train, data.model_values, np.nan)
    scale = np.nanstd(observed, axis=0)
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > EPS)])
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback if np.isfinite(fallback) else 1.0)
    feature_gap = 1.0 - train.mean(axis=0)
    row_gap = 1.0 - train.mean(axis=1)
    disagreement = np.abs(pred - legacy.stat_imputer(data, train, "median"))
    return np.maximum(0.45 * disagreement + scale[None, :] * (0.35 + feature_gap[None, :] + 0.20 * row_gap[:, None]), 1e-4)


def fast_knn_imputer(data: base.PreparedData, train: np.ndarray, k: int = 10, candidate_pool: int = 64) -> np.ndarray:
    median = legacy.stat_imputer(data, train, "median")
    filled = np.where(train, data.model_values, median)
    filled = np.nan_to_num(filled, nan=0.0)
    norms = np.sum(filled * filled, axis=1)
    dist = norms[:, None] + norms[None, :] - 2.0 * (filled @ filled.T)
    dist = np.maximum(dist, 0.0)
    np.fill_diagonal(dist, np.inf)
    pool = min(candidate_pool, max(1, dist.shape[0] - 1))
    nearest = np.argpartition(dist, kth=pool - 1, axis=1)[:, :pool]
    nearest_dist = np.take_along_axis(dist, nearest, axis=1)
    order = np.argsort(nearest_dist, axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_dist = np.take_along_axis(nearest_dist, order, axis=1)
    pred = median.copy()
    base_weights = 1.0 / np.maximum(nearest_dist, 1e-6)
    for j in range(data.model_values.shape[1]):
        values = data.model_values[:, j]
        valid = train[:, j][nearest]
        weights = base_weights * valid
        if k < weights.shape[1]:
            weights[:, k:] = 0.0
        denom = weights.sum(axis=1)
        estimate = (weights * values[nearest]).sum(axis=1) / np.maximum(denom, 1e-12)
        pred[:, j] = np.where(denom > 1e-12, estimate, median[:, j])
    return np.where(np.isfinite(pred), pred, median)


def build_candidates(data: base.PreparedData, train: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    median = legacy.stat_imputer(data, train, "median")
    candidates = {
        "Median": median,
        "KNN": fast_knn_imputer(data, train, k=10),
        "SoftImpute": legacy.softimpute(
            data,
            train,
            rank=min(12, max(3, data.model_values.shape[1] // 4)),
            shrink=0.18,
            epochs=10,
        ),
        "Low-rank SVD": base.matrix_factorization(
            data,
            train,
            rank=min(10, max(3, data.model_values.shape[1] // 5)),
            epochs=10,
            seed=seed,
            use_evidence=False,
        ),
    }
    return {name: np.where(np.isfinite(pred), pred, median) for name, pred in candidates.items()}


def candidate_stack(candidates: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([candidates[name] for name in EXPERT_METHODS], axis=0)


def dual_stack_predictions(
    data: base.PreparedData,
    train: np.ndarray,
    seed: int,
    final_candidates: dict[str, np.ndarray] | None = None,
    use_cross_fitting: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + 3301)
    eligible = train.copy()
    eligible[:, train.sum(axis=0) < 20] = False
    validation = eligible & (rng.random(eligible.shape) < 0.12)
    if validation.sum() < 280:
        validation = eligible & (rng.random(eligible.shape) < 0.18)
    fit_mask = train & ~validation if use_cross_fitting else train
    fit_candidates = build_candidates(data, fit_mask, seed + 7)
    final_candidates = final_candidates if final_candidates is not None else build_candidates(data, train, seed + 29)
    fit_stack = candidate_stack(fit_candidates)
    final_stack = candidate_stack(final_candidates)
    groups = np.array([base.group_id(feature) for feature in data.features])
    priors = {
        "Ours-L1": np.array([0.30, 0.25, 0.25, 0.20]),
        "Ours-L2": np.array([0.12, 0.28, 0.36, 0.24]),
    }
    weight_fns = {
        "Ours-L1": legacy.robust_convex_weights,
        "Ours-L2": legacy.convex_weights,
    }
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fallback = final_candidates["Median"]
    for method in ["Ours-L1", "Ours-L2"]:
        weight_fn = weight_fns[method]
        global_w = weight_fn(
            fit_stack[:, validation].T,
            data.model_values[validation],
            prior=priors[method],
            ridge=0.10,
        )
        group_weights: dict[str, np.ndarray] = {}
        residual_scale = np.full(len(data.features), np.nan, dtype=float)
        for group in sorted(set(groups)):
            group_validation = validation & (groups == group)[None, :]
            if group_validation.sum() >= 36:
                w = weight_fn(
                    fit_stack[:, group_validation].T,
                    data.model_values[group_validation],
                    prior=global_w,
                    ridge=0.14,
                )
            else:
                w = global_w
            group_weights[group] = w
            for j in np.where(groups == group)[0]:
                cell_mask = validation[:, j]
                if cell_mask.sum() >= 10:
                    err = fit_stack[:, cell_mask, j].T @ w - data.model_values[cell_mask, j]
                    residual_scale[j] = float(np.median(np.abs(err - np.median(err))))
        fallback_scale = float(np.nanmedian(residual_scale[np.isfinite(residual_scale)]))
        residual_scale = np.where(np.isfinite(residual_scale) & (residual_scale > 1e-4), residual_scale, max(fallback_scale, 0.08))
        pred = np.zeros_like(data.model_values)
        for j, group in enumerate(groups):
            pred[:, j] = final_stack[:, :, j].T @ group_weights[group]
        disagreement = np.std(final_stack, axis=0)
        missingness = 1.0 - train.mean(axis=0)
        unc = np.maximum(0.58 * disagreement + 0.30 * residual_scale[None, :] + 0.12 * missingness[None, :], 1e-4)
        outputs[method] = (np.where(np.isfinite(pred), pred, fallback), unc)
    return outputs


def violation(data: base.PreparedData, pred: np.ndarray) -> dict[str, float]:
    metrics = legacy.constraint_metrics(data, pred)
    pieces = [
        metrics.get("range_violation_magnitude", np.nan),
        metrics.get("composition_sum_abs_error", np.nan),
        metrics.get("treatment_sum_abs_error", np.nan),
    ]
    metrics["violation"] = float(sum(x for x in pieces if np.isfinite(x)))
    return metrics


def metric_row(data: base.PreparedData, pred: np.ndarray, test: np.ndarray, protocol: str, method: str, seed: int) -> dict[str, object]:
    row: dict[str, object] = base.evaluate(pred, data.model_values, test, None)
    row["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
    row.update(violation(data, pred))
    row.update({"protocol": protocol, "method": method, "seed": seed})
    return row


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [c for c in frame.columns if c not in set(keys + ["seed"]) and pd.api.types.is_numeric_dtype(frame[c])]
    grouped = frame.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().reset_index()
    std = grouped[numeric].std(ddof=1).add_suffix("_std").reset_index()
    std = std.rename(columns={f"{key}_std": key for key in keys})
    return mean.merge(std, on=keys, how="left").merge(grouped.size().rename("seeds").reset_index(), on=keys, how="left")


def protocol_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baselines = [m for m in METHODS if not m.startswith("Ours")]
    for protocol in sorted(raw["protocol"].unique()):
        sub = raw[raw["protocol"].eq(protocol)]
        means = sub.groupby("method")[["mae_norm", "rmse_norm", "violation"]].mean()
        best_mae_method = means.loc[baselines, "mae_norm"].idxmin()
        best_rmse_method = means.loc[baselines, "rmse_norm"].idxmin()
        l1 = means.loc["Ours-L1"]
        l2 = means.loc["Ours-L2"]
        paired_l1 = (
            sub[sub.method.eq("Ours-L1")].set_index("seed")["mae_norm"]
            - sub[sub.method.eq(best_mae_method)].set_index("seed")["mae_norm"]
        ).dropna()
        paired_l2 = (
            sub[sub.method.eq("Ours-L2")].set_index("seed")["rmse_norm"]
            - sub[sub.method.eq(best_rmse_method)].set_index("seed")["rmse_norm"]
        ).dropna()
        rows.append(
            {
                "protocol": protocol,
                "ours_l1_mae": float(l1.mae_norm),
                "best_baseline_mae_method": best_mae_method,
                "best_baseline_mae": float(means.loc[best_mae_method, "mae_norm"]),
                "delta_mae": float(l1.mae_norm - means.loc[best_mae_method, "mae_norm"]),
                "mae_wins": int((paired_l1 < -EPS).sum()),
                "mae_losses": int((paired_l1 > EPS).sum()),
                "ours_l2_rmse": float(l2.rmse_norm),
                "best_baseline_rmse_method": best_rmse_method,
                "best_baseline_rmse": float(means.loc[best_rmse_method, "rmse_norm"]),
                "delta_rmse": float(l2.rmse_norm - means.loc[best_rmse_method, "rmse_norm"]),
                "rmse_wins": int((paired_l2 < -EPS).sum()),
                "rmse_losses": int((paired_l2 > EPS).sum()),
                "ours_l1_violation": float(l1.violation),
                "ours_l2_violation": float(l2.violation),
                "seeds": int(sub["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(data: base.PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        masks = protocol_masks(data, seed)
        for protocol, test in masks.items():
            if protocol != "random_20pct" and seed not in STRUCTURED_SEEDS:
                continue
            train = data.observed_mask & ~test
            for i, j in np.argwhere(test):
                mask_rows.append({"seed": seed, "protocol": protocol, "city_code": data.city.iloc[i]["city_code"], "feature": data.features[j]})
            candidates = build_candidates(data, train, seed + 17)
            stacked = dual_stack_predictions(data, train, seed + 37, candidates)
            raw_predictions = {**candidates, **{name: value[0] for name, value in stacked.items()}}
            for method in METHODS:
                raw = raw_predictions[method]
                projected = legacy.exact_feasibility_projection(data, raw)
                rows.append(metric_row(data, projected, test, protocol, method, seed))
    raw_results = pd.DataFrame(rows)
    masks_df = pd.DataFrame(mask_rows)
    raw_results.to_csv(TABLE_DIR / "defra_external_by_seed.csv", index=False, encoding="utf-8-sig")
    aggregate(raw_results, ["protocol", "method"]).to_csv(TABLE_DIR / "defra_external_summary.csv", index=False, encoding="utf-8-sig")
    protocol_summary(raw_results).to_csv(TABLE_DIR / "defra_external_protocol_summary.csv", index=False, encoding="utf-8-sig")
    masks_df.to_csv(PROTOCOL_DIR / "defra_external_shared_masks.csv", index=False, encoding="utf-8-sig")
    return raw_results, masks_df


def run_ablation(data: base.PreparedData) -> pd.DataFrame:
    variants: list[tuple[str, str, dict[str, object]]] = [
        ("Full Ours-L1", "l1", {}),
        ("SoftImpute + projection", "softimpute", {}),
        ("Ours-L2 only", "l2", {}),
        ("Unweighted projection", "l1", {"projection_mode": "unweighted"}),
        ("w/o cross-fitting", "l1", {"use_cross_fitting": False}),
        ("w/o projection", "l1", {"use_constraints": False}),
    ]
    rows: list[dict[str, object]] = []
    for seed in STRUCTURED_SEEDS:
        test = random_mask(data, seed + 2200)
        train = data.observed_mask & ~test
        candidates = build_candidates(data, train, seed + 2310)
        full = dual_stack_predictions(data, train, seed + 2320, candidates, use_cross_fitting=True)
        leaked = dual_stack_predictions(data, train, seed + 2330, candidates, use_cross_fitting=False)
        for variant, head, options in variants:
            if head == "l2":
                raw = full["Ours-L2"][0]
                pred = legacy.exact_feasibility_projection(data, raw)
            elif head == "softimpute":
                raw = candidates["SoftImpute"]
                pred = legacy.exact_feasibility_projection(data, raw)
            else:
                if variant == "w/o cross-fitting":
                    raw = leaked["Ours-L1"][0]
                else:
                    raw = full["Ours-L1"][0]
                if options.get("use_constraints") is False:
                    pred = raw
                elif options.get("projection_mode") == "unweighted":
                    pred = legacy.unweighted_feasibility_projection(data, raw)
                else:
                    pred = legacy.exact_feasibility_projection(data, raw)
            row: dict[str, object] = base.evaluate(pred, data.model_values, test, None)
            row["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
            row.update(violation(data, pred))
            row.update({"variant": variant, "seed": seed})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "defra_external_ablation_by_seed.csv", index=False, encoding="utf-8-sig")
    aggregate(out, ["variant"]).to_csv(TABLE_DIR / "defra_external_ablation_summary.csv", index=False, encoding="utf-8-sig")
    return out


def run_acquisition(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in ACQUISITION_SEEDS:
        initial_test = base.make_random_mask(data, 0.16, seed + 6100)
        for strategy in ["Ours safe", "Random"]:
            rng = np.random.default_rng(seed + (7100 if strategy == "Random" else 8100))
            train = data.observed_mask & ~initial_test
            remaining = initial_test.copy()
            for round_idx in range(4):
                candidates = build_candidates(data, train, seed + 9000 + round_idx)
                raw, unc = dual_stack_predictions(data, train, seed + 9100 + round_idx, candidates)["Ours-L1"]
                pred = legacy.exact_feasibility_projection(data, raw)
                metric = base.evaluate(pred, data.model_values, remaining, None)
                metric.update({"strategy": strategy, "seed": seed, "round": round_idx, "cumulative_budget": int(round_idx * 20), "remaining_cells": int(remaining.sum())})
                rows.append(metric)
                if round_idx == 3 or remaining.sum() == 0:
                    continue
                entries = np.argwhere(remaining)
                if strategy == "Ours safe":
                    scores = unc[remaining]
                    order = np.argsort(scores)[::-1]
                else:
                    order = rng.permutation(len(entries))
                chosen = entries[order[: min(20, len(entries))]]
                train[chosen[:, 0], chosen[:, 1]] = True
                remaining[chosen[:, 0], chosen[:, 1]] = False
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "defra_external_acquisition_by_seed.csv", index=False, encoding="utf-8-sig")
    aggregate(out, ["strategy", "cumulative_budget"]).to_csv(TABLE_DIR / "defra_external_acquisition_summary.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-acquisition", action="store_true", help="Skip reveal-refit external acquisition simulation.")
    parser.add_argument("--only-acquisition", action="store_true", help="Only run the Defra reveal-refit acquisition simulation.")
    args = parser.parse_args()
    ensure_dirs()
    data, summary = load_defra_data()
    if args.only_acquisition:
        acquisition = run_acquisition(data)
        metadata = {
            "dataset": summary.iloc[0].to_dict(),
            "acquisition_rows": int(len(acquisition)),
            "acquisition_seeds": ACQUISITION_SEEDS,
        }
        (OUT_DIR / "defra_external_acquisition_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return
    benchmark, masks = run_benchmark(data)
    ablation = run_ablation(data)
    acquisition_rows = 0
    if not args.skip_acquisition:
        acquisition_rows = int(len(run_acquisition(data)))
    metadata = {
        "dataset": summary.iloc[0].to_dict(),
        "benchmark_rows": int(len(benchmark)),
        "mask_rows": int(len(masks)),
        "ablation_rows": int(len(ablation)),
        "acquisition_rows": acquisition_rows,
        "methods": METHODS,
        "random_seeds": SEEDS,
        "structured_seeds": STRUCTURED_SEEDS,
    }
    (OUT_DIR / "defra_external_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
