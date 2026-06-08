from __future__ import annotations

import argparse
import json
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
import run_experiment as legacy
import run_longterm_experiment as longterm
import run_aaai_experiments as aaai


OUT_DIR = ROOT / "outputs" / "ecowaste_longterm"
TABLE_DIR = OUT_DIR / "tables"
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
STRUCTURED_SEEDS = SEEDS[:5]
ITERATIVE_SEEDS = SEEDS[:3]
EPS = 1e-9


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return aaai.aggregate(frame, keys)


def metric_row(data: base.PreparedData, pred: np.ndarray, test: np.ndarray, **labels: object) -> dict[str, object]:
    row: dict[str, object] = base.evaluate(pred, data.model_values, test, None)
    row["violation"] = aaai.violation(data, pred)
    row["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
    row.update(labels)
    return row


def tune_dae(data: base.PreparedData, train: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed + 9101)
    eligible = train.copy()
    eligible[:, train.sum(axis=0) < 18] = False
    validation = eligible & (rng.random(eligible.shape) < 0.10)
    fit = train & ~validation
    candidates = [
        {"hidden": 28, "noise": 0.12, "lr": 0.008, "epochs": 70},
        {"hidden": 44, "noise": 0.18, "lr": 0.010, "epochs": 80},
        {"hidden": 64, "noise": 0.25, "lr": 0.006, "epochs": 90},
    ]
    records = []
    for index, params in enumerate(candidates):
        pred = legacy.denoising_autoencoder_imputer(data, fit, seed=seed + 100 * index, **params)
        mae = float(np.mean(np.abs(pred[validation] - data.model_values[validation])))
        rmse = float(np.sqrt(np.mean((pred[validation] - data.model_values[validation]) ** 2)))
        records.append({**params, "validation_mae": mae, "validation_rmse": rmse})
    chosen = min(records, key=lambda row: (row["validation_mae"], row["validation_rmse"]))
    params = {key: chosen[key] for key in ["hidden", "noise", "lr", "epochs"]}
    final = legacy.denoising_autoencoder_imputer(data, train, seed=seed + 999, **params)
    return final, chosen


def deep_baseline(data: base.PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tuning: list[dict[str, object]] = []
    for seed in SEEDS:
        test = base.make_random_mask(data, 0.20, seed)
        train = data.observed_mask & ~test
        dae, choice = tune_dae(data, train, seed)
        dae = legacy.exact_feasibility_projection(data, dae)
        rows.append(metric_row(data, dae, test, method="Tuned DAE", seed=seed))
        tuning.append({"seed": seed, **choice})

        gain = legacy.gain_style_imputer(data, train, seed=seed + 71)
        gain = legacy.exact_feasibility_projection(data, gain)
        rows.append(metric_row(data, gain, test, method="GAIN-style", seed=seed))
    return pd.DataFrame(rows), pd.DataFrame(tuning)


def structured_benchmark(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = ["Median", "SoftImpute", "MissForest", "Ours-L1", "Ours-L2"]
    for seed in STRUCTURED_SEEDS:
        for scenario, test in longterm.scenario_masks(data, seed).items():
            train = data.observed_mask & ~test
            for method in methods:
                if method == "Median":
                    raw = legacy.stat_imputer(data, train, "median")
                elif method == "SoftImpute":
                    raw = legacy.softimpute(data, train, rank=18, shrink=0.16, epochs=32)
                elif method == "MissForest":
                    raw = legacy.missforest_imputer(data, train, seed=seed + 51, iterations=2, trees=5)
                elif method == "Ours-L1":
                    raw, _ = legacy.safe_pccm_predict(data, train, seed=seed + 67, use_evidence=False, use_constraints=False)
                elif method == "Ours-L2":
                    raw, _ = legacy.pccm_l2_predict(data, train, seed=seed + 79, use_constraints=False)
                else:
                    raise ValueError(method)
                pred = legacy.exact_feasibility_projection(data, raw)
                rows.append(metric_row(data, pred, test, scenario=scenario, method=method, seed=seed))
    return pd.DataFrame(rows)


def acquisition_scores(
    data: base.PreparedData,
    train: np.ndarray,
    remaining: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if strategy == "Random":
        return rng.random(int(remaining.sum()))
    if strategy == "Uncertainty-only":
        return uncertainty[remaining]
    feature_gap = 1.0 - train.mean(axis=0)
    median = legacy.stat_imputer(data, train, "median")
    disagreement = np.abs(pred - median)
    score = uncertainty * (1.0 + 0.45 * feature_gap[None, :]) + 0.20 * disagreement
    return score[remaining]


def iterative_verification(data: base.PreparedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategies = ["Random", "Uncertainty-only", "Learned no-evidence", "Ours safe"]
    row_missing = 1.0 - data.observed_mask.mean(axis=1)
    col_missing = 1.0 - data.observed_mask.mean(axis=0)
    groups = np.array([base.group_id(feature) for feature in data.features])
    costly_groups = {"treatment", "collection", "budget", "institutional", "legal", "uncollected"}

    def cost_profile(selected: np.ndarray) -> dict[str, float]:
        if len(selected) == 0:
            return {"uniform": 0.0, "field": 0.0, "city": 0.0}
        rows_idx = selected[:, 0]
        cols_idx = selected[:, 1]
        group_cost = np.array([0.75 if groups[j] in costly_groups else 0.0 for j in cols_idx], dtype=float)
        return {
            "uniform": float(len(selected)),
            "field": float(np.sum(1.0 + 0.75 * col_missing[cols_idx] + group_cost)),
            "city": float(np.sum(1.0 + 1.50 * row_missing[rows_idx])),
        }

    for seed in ITERATIVE_SEEDS:
        initial_test = base.make_random_mask(data, 0.20, seed + 6200)
        for strategy_index, strategy in enumerate(strategies):
            train = data.observed_mask & ~initial_test
            remaining = initial_test.copy()
            rng = np.random.default_rng(seed + 6300 + strategy_index)
            cumulative_costs = {"uniform": 0.0, "field": 0.0, "city": 0.0}
            for round_id in range(4):
                use_evidence = strategy == "Ours safe"
                pred, uncertainty = legacy.safe_pccm_predict(
                    data,
                    train,
                    seed=seed + 6400 + round_id,
                    use_evidence=use_evidence,
                    use_constraints=True,
                )
                result = metric_row(
                    data,
                    pred,
                    remaining,
                    strategy=strategy,
                    seed=seed,
                    round=round_id,
                    cumulative_budget=20 * round_id,
                    remaining_cells=int(remaining.sum()),
                    cumulative_cost_uniform=cumulative_costs["uniform"],
                    cumulative_cost_field=cumulative_costs["field"],
                    cumulative_cost_city=cumulative_costs["city"],
                )
                rows.append(result)
                if round_id == 3 or remaining.sum() == 0:
                    continue
                entries = np.argwhere(remaining)
                scores = acquisition_scores(data, train, remaining, pred, uncertainty, strategy, rng)
                selected_index = np.argsort(scores)[::-1][: min(20, len(entries))]
                selected = entries[selected_index]
                selected_costs = cost_profile(selected)
                for key, value in selected_costs.items():
                    cumulative_costs[key] += value
                train[selected[:, 0], selected[:, 1]] = True
                remaining[selected[:, 0], selected[:, 1]] = False
    return pd.DataFrame(rows)


def cost_aware_summary(iterative: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    terminal = iterative.sort_values("round").groupby(["strategy", "seed"], as_index=False).tail(1)
    initial = iterative[iterative["round"] == 0][["strategy", "seed", "mae_norm", "rmse_norm"]].rename(
        columns={"mae_norm": "initial_mae_norm", "rmse_norm": "initial_rmse_norm"}
    )
    joined = terminal.merge(initial, on=["strategy", "seed"], how="left")
    rows: list[dict[str, object]] = []
    for profile, column in [
        ("uniform cell cost", "cumulative_cost_uniform"),
        ("field-dependent cost", "cumulative_cost_field"),
        ("city-monitoring cost", "cumulative_cost_city"),
    ]:
        for row in joined.itertuples(index=False):
            cost = float(getattr(row, column))
            mae_gain = float(row.initial_mae_norm - row.mae_norm)
            rmse_gain = float(row.initial_rmse_norm - row.rmse_norm)
            rows.append(
                {
                    "strategy": row.strategy,
                    "seed": row.seed,
                    "cost_profile": profile,
                    "spent_cost": cost,
                    "terminal_mae_norm": float(row.mae_norm),
                    "terminal_rmse_norm": float(row.rmse_norm),
                    "mae_reduction": mae_gain,
                    "rmse_reduction": rmse_gain,
                    "mae_reduction_per_cost": mae_gain / max(cost, EPS),
                    "rmse_reduction_per_cost": rmse_gain / max(cost, EPS),
                }
            )
    by_seed = pd.DataFrame(rows)
    return by_seed, aggregate(by_seed, ["strategy", "cost_profile"])


def decision_depth_summary() -> pd.DataFrame:
    city = pd.read_csv(TABLE_DIR / "city_strong_benchmark_summary.csv", encoding="utf-8-sig").set_index("method")
    rows = [
        {
            "policy": "Dual-risk delivery",
            "mae_head": "Ours-L1",
            "rmse_head": "Ours-L2",
            "mae_norm": city.loc["Ours-L1", "mae_norm"],
            "rmse_norm": city.loc["Ours-L2", "rmse_norm"],
        },
        {
            "policy": "Single L1 head for both losses",
            "mae_head": "Ours-L1",
            "rmse_head": "Ours-L1",
            "mae_norm": city.loc["Ours-L1", "mae_norm"],
            "rmse_norm": city.loc["Ours-L1", "rmse_norm"],
        },
        {
            "policy": "Single L2 head for both losses",
            "mae_head": "Ours-L2",
            "rmse_head": "Ours-L2",
            "mae_norm": city.loc["Ours-L2", "mae_norm"],
            "rmse_norm": city.loc["Ours-L2", "rmse_norm"],
        },
        {
            "policy": "Strong low-rank baseline",
            "mae_head": "SoftImpute",
            "rmse_head": "SoftImpute",
            "mae_norm": city.loc["SoftImpute", "mae_norm"],
            "rmse_norm": city.loc["SoftImpute", "rmse_norm"],
        },
    ]
    return pd.DataFrame(rows)


def structured_protocol_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate(frame, ["scenario", "method"])
    rows: list[dict[str, object]] = []
    for scenario, group in summary.groupby("scenario"):
        l1 = group[group.method == "Ours-L1"].iloc[0]
        l2 = group[group.method == "Ours-L2"].iloc[0]
        baselines = group[~group.method.str.startswith("Ours")]
        best_mae = baselines.sort_values("mae_norm").iloc[0]
        best_rmse = baselines.sort_values("rmse_norm").iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "ours_l1_mae": l1.mae_norm,
                "best_baseline_mae_method": best_mae.method,
                "best_baseline_mae": best_mae.mae_norm,
                "mae_delta": l1.mae_norm - best_mae.mae_norm,
                "ours_l2_rmse": l2.rmse_norm,
                "best_baseline_rmse_method": best_rmse.method,
                "best_baseline_rmse": best_rmse.rmse_norm,
                "rmse_delta": l2.rmse_norm - best_rmse.rmse_norm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "deep", "structured", "iterative", "derived"], default="all")
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)

    if args.stage in {"all", "deep"}:
        deep, tuning = deep_baseline(data)
        deep.to_csv(TABLE_DIR / "formal_deep_baseline_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(deep, ["method"]).to_csv(TABLE_DIR / "formal_deep_baseline_summary.csv", index=False, encoding="utf-8-sig")
        tuning.to_csv(TABLE_DIR / "formal_deep_baseline_tuning.csv", index=False, encoding="utf-8-sig")

    if args.stage in {"all", "structured"}:
        structured = structured_benchmark(data)
        structured.to_csv(TABLE_DIR / "city_structured_benchmark_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(structured, ["scenario", "method"]).to_csv(
            TABLE_DIR / "city_structured_benchmark_summary.csv", index=False, encoding="utf-8-sig"
        )
        structured_protocol_summary(structured).to_csv(
            TABLE_DIR / "city_structured_protocol_summary.csv", index=False, encoding="utf-8-sig"
        )

    if args.stage in {"all", "iterative"}:
        iterative = iterative_verification(data)
        iterative.to_csv(TABLE_DIR / "iterative_verification_by_seed.csv", index=False, encoding="utf-8-sig")
        aggregate(iterative, ["strategy", "round", "cumulative_budget"]).to_csv(
            TABLE_DIR / "iterative_verification_summary.csv", index=False, encoding="utf-8-sig"
        )
        cost_by_seed, cost_summary = cost_aware_summary(iterative)
        cost_by_seed.to_csv(TABLE_DIR / "cost_aware_verification_by_seed.csv", index=False, encoding="utf-8-sig")
        cost_summary.to_csv(TABLE_DIR / "cost_aware_verification_summary.csv", index=False, encoding="utf-8-sig")

    if args.stage in {"all", "derived"}:
        decision_depth_summary().to_csv(TABLE_DIR / "dual_risk_decision_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "technical_depth_additions": [
            "validation-tuned denoising autoencoder baseline",
            "five-protocol city structured-missingness benchmark",
            "three-round reveal-refit-reevaluate verification simulation",
            "cost-aware verification summary under uniform, field-dependent, and city-monitoring cost profiles",
            "dual-risk decision-value summary",
        ],
        "external_validation": "Official third-party data download unavailable in the current network environment; no synthetic external claim is made.",
    }
    (OUT_DIR / "technical_depth_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
