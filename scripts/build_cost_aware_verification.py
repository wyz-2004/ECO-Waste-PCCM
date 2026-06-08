from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = Path(os.environ["ECOWASTE_PROJECT_ROOT"]).resolve() if os.environ.get("ECOWASTE_PROJECT_ROOT") else SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import ecowaste_core as base
import run_aaai_experiments as aaai


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
EPS = 1e-9


def candidate_cost_multipliers(data: base.PreparedData) -> dict[str, float]:
    missing = ~data.observed_mask
    row_missing = 1.0 - data.observed_mask.mean(axis=1)
    col_missing = 1.0 - data.observed_mask.mean(axis=0)
    groups = np.array([base.group_id(feature) for feature in data.features])
    costly_groups = {"treatment", "collection", "budget", "institutional", "legal", "uncollected"}

    row_idx, col_idx = np.where(missing)
    if len(row_idx) == 0:
        return {
            "uniform cell cost": 1.0,
            "field-dependent cost": 1.0,
            "city-monitoring cost": 1.0,
        }
    group_cost = np.array([0.75 if groups[j] in costly_groups else 0.0 for j in col_idx], dtype=float)
    field_cost = 1.0 + 0.75 * col_missing[col_idx] + group_cost
    city_cost = 1.0 + 1.50 * row_missing[row_idx]
    return {
        "uniform cell cost": 1.0,
        "field-dependent cost": float(np.mean(field_cost)),
        "city-monitoring cost": float(np.mean(city_cost)),
    }


def build_cost_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(TABLE_DIR / "iterative_verification_by_seed.csv", encoding="utf-8-sig")
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)
    multipliers = candidate_cost_multipliers(data)

    initial = source[source["round"] == 0][["strategy", "seed", "mae_norm", "rmse_norm"]].rename(
        columns={"mae_norm": "initial_mae_norm", "rmse_norm": "initial_rmse_norm"}
    )
    joined = source.merge(initial, on=["strategy", "seed"], how="left")
    rows: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        for profile, multiplier in multipliers.items():
            cost = float(row.cumulative_budget) * multiplier
            mae_reduction = float(row.initial_mae_norm - row.mae_norm)
            rmse_reduction = float(row.initial_rmse_norm - row.rmse_norm)
            rows.append(
                {
                    "strategy": row.strategy,
                    "seed": int(row.seed),
                    "round": int(row.round),
                    "cumulative_budget": int(row.cumulative_budget),
                    "cost_profile": profile,
                    "cost_multiplier": multiplier,
                    "simulated_cost": cost,
                    "mae_norm": float(row.mae_norm),
                    "rmse_norm": float(row.rmse_norm),
                    "mae_reduction": mae_reduction,
                    "rmse_reduction": rmse_reduction,
                    "mae_reduction_per_cost": mae_reduction / max(cost, EPS),
                    "rmse_reduction_per_cost": rmse_reduction / max(cost, EPS),
                }
            )
    by_seed = pd.DataFrame(rows)
    summary = aaai.aggregate(by_seed, ["strategy", "round", "cumulative_budget", "cost_profile"])
    return by_seed, summary


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_seed, summary = build_cost_tables()
    by_seed.to_csv(TABLE_DIR / "cost_aware_verification_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "cost_aware_verification_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {TABLE_DIR / 'cost_aware_verification_by_seed.csv'}")
    print(f"Wrote {TABLE_DIR / 'cost_aware_verification_summary.csv'}")


if __name__ == "__main__":
    main()
