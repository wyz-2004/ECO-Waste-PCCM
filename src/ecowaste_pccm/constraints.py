from __future__ import annotations

import numpy as np

from .data import feature_bounds


COMPOSITION_GROUP = "composition_msw_"
SEPARATION_GROUP = "separation_breakdown_"
TREATMENT_FEATURES = [
    "waste_treatment_open_dumpsite_percent",
    "waste_treatment_controlled_landfill_percent",
    "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
    "waste_treatment_landfill_unspecified_percent",
    "waste_treatment_anaerobic_digestion_percent",
    "waste_treatment_compost_percent",
    "waste_treatment_recycling_percent",
    "waste_treatment_incineration_percent",
    "waste_treatment_mbt_percent",
    "waste_treatment_rdf_percent",
    "waste_treatment_other_percent",
    "waste_uncollected_percent",
    "waste_treatment_unaccounted_for_percent",
]


def closure_groups(features: list[str], values: np.ndarray) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    comp = [i for i, f in enumerate(features) if f.startswith(COMPOSITION_GROUP) and f.endswith("_percent")]
    sep = [i for i, f in enumerate(features) if f.startswith(SEPARATION_GROUP) and f.endswith("_percent")]
    treat = [i for i, f in enumerate(features) if f in TREATMENT_FEATURES]
    if len(comp) >= 3:
        groups["composition_share_closure"] = comp
    if len(sep) >= 3:
        groups["separation_share_closure"] = sep
    if len(treat) >= 3:
        groups["treatment_share_closure"] = treat
    return groups


def _target_for_group(values: np.ndarray, idxs: list[int]) -> float:
    observed = values[:, idxs]
    max_obs = float(np.nanmax(observed)) if np.isfinite(observed).any() else 1.0
    return 1.0 if max_obs <= 1.5 else 100.0


def project_workbook(values: np.ndarray, features: list[str], reference_values: np.ndarray) -> np.ndarray:
    projected = values.copy()
    for j, feature in enumerate(features):
        observed_values = reference_values[:, j]
        lo, hi = feature_bounds(feature, observed_values)
        if lo is not None:
            projected[:, j] = np.maximum(projected[:, j], lo)
        if hi is not None:
            projected[:, j] = np.minimum(projected[:, j], hi)

    for _, idxs in closure_groups(features, reference_values).items():
        target = _target_for_group(reference_values, idxs)
        block = np.maximum(projected[:, idxs], 0.0)
        totals = block.sum(axis=1)
        valid = totals > 1e-12
        block[valid] = block[valid] / totals[valid, None] * target
        if (~valid).any():
            block[~valid] = target / len(idxs)
        projected[:, idxs] = block
    return projected


def closure_group_for_feature(feature: str, features: list[str], reference_values: np.ndarray) -> str:
    for name, idxs in closure_groups(features, reference_values).items():
        if feature in [features[i] for i in idxs]:
            return name
    return ""


def closure_violation(values: np.ndarray, features: list[str], reference_values: np.ndarray) -> float:
    groups = closure_groups(features, reference_values)
    if not groups:
        return 0.0
    violations = []
    for _, idxs in groups.items():
        target = _target_for_group(reference_values, idxs)
        totals = np.nansum(values[:, idxs], axis=1)
        violations.extend(np.abs(totals - target).tolist())
    return float(np.mean(violations)) if violations else 0.0
