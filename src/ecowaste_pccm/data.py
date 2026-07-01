from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = [
    "country_code",
    "iso3c",
    "region_id",
    "country_name",
    "income_id",
    "income_id_2022",
    "city_name",
    "city_code",
]


@dataclass
class WorkbookData:
    values: np.ndarray
    observed: np.ndarray
    features: list[str]
    metadata: pd.DataFrame
    raw_frame: pd.DataFrame


def _deduplicate_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        key = str(col).strip()
        if key in seen:
            seen[key] += 1
            key = f"{key}.{seen[key]}"
        else:
            seen[key] = 0
        out.append(key)
    return out


def load_city_workbook(
    path: Path,
    min_observed_per_feature: int = 8,
    feature_prefixes: list[str] | None = None,
) -> WorkbookData:
    frame = pd.read_excel(path, sheet_name="City dataset")
    frame.columns = _deduplicate_columns(list(frame.columns))

    id_cols = [c for c in ID_COLUMNS if c in frame.columns]
    metadata = frame[id_cols].copy()
    if "city_name" not in metadata.columns:
        metadata["city_name"] = [f"city_{i}" for i in range(len(frame))]
    if "city_code" not in metadata.columns:
        metadata["city_code"] = metadata["city_name"].astype(str).str.lower().str.replace(" ", "_")

    numeric_cols: list[str] = []
    numeric_values: dict[str, pd.Series] = {}
    for col in frame.columns:
        if col in id_cols:
            continue
        if feature_prefixes and not any(col.startswith(prefix) for prefix in feature_prefixes):
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if int(series.notna().sum()) >= min_observed_per_feature:
            name = col.lower()
            finite = series[np.isfinite(series)]
            if (
                ("percent" in name or "share" in name or "coverage" in name)
                and len(finite)
                and float(finite.max()) > 1.5
            ):
                series = series / 100.0
            numeric_cols.append(col)
            numeric_values[col] = series

    if not numeric_cols:
        raise ValueError("No numeric workbook columns met the observation threshold.")

    numeric = pd.DataFrame(numeric_values)
    values = numeric.to_numpy(dtype=float)
    observed = np.isfinite(values)

    return WorkbookData(
        values=values,
        observed=observed,
        features=numeric_cols,
        metadata=metadata.reset_index(drop=True),
        raw_frame=frame.reset_index(drop=True),
    )


def feature_kind(feature: str, observed_values: np.ndarray) -> str:
    name = feature.lower()
    finite = observed_values[np.isfinite(observed_values)]
    if "percent" in name or "share" in name or "coverage" in name:
        return "rate"
    if (
        "number" in name
        or "population" in name
        or "tons" in name
        or "kg_per_cap" in name
        or "amount" in name
        or "usd" in name
        or "capacity" in name
        or "distance" in name
    ):
        return "amount"
    if name.endswith("_year") or "date" in name:
        return "year"
    if len(finite) and float(np.nanmin(finite)) >= 0:
        return "nonnegative"
    return "numeric"


def feature_bounds(feature: str, observed_values: np.ndarray) -> tuple[float | None, float | None]:
    kind = feature_kind(feature, observed_values)
    finite = observed_values[np.isfinite(observed_values)]
    if kind == "rate":
        max_obs = float(np.nanmax(finite)) if len(finite) else 1.0
        return (0.0, 1.0 if max_obs <= 1.5 else 100.0)
    if kind in {"amount", "nonnegative"}:
        return (0.0, None)
    return (None, None)


def group_keys(metadata: pd.DataFrame) -> list[tuple[str, str]]:
    if "region_id" in metadata:
        region = metadata["region_id"].fillna("unknown").astype(str).tolist()
    else:
        region = ["unknown"] * len(metadata)
    if "income_id" in metadata:
        income = metadata["income_id"].fillna("unknown").astype(str).tolist()
    else:
        income = ["unknown"] * len(metadata)
    return list(zip(region, income))
