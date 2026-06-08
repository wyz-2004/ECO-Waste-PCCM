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
import run_experiment as legacy


TABLE_DIR = ROOT / "outputs" / "ecowaste_longterm" / "tables"
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [c for c in frame.columns if c not in set(keys + ["seed"]) and pd.api.types.is_numeric_dtype(frame[c])]
    grouped = frame.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().reset_index()
    std = grouped[numeric].std(ddof=1).add_suffix("_std").reset_index()
    std = std.rename(columns={f"{key}_std": key for key in keys})
    return mean.merge(std, on=keys, how="left").merge(grouped.size().rename("seeds").reset_index(), on=keys, how="left")


def bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260607)
    rows = []
    for head in ["PCCM-L1+Proj", "PCCM-L2+Proj"]:
        for baseline in ["Median+Proj", "SoftImpute+Proj", "Low-rank SVD+Proj"]:
            for metric in ["mae_norm", "rmse_norm"]:
                left = frame[frame.method == head].set_index("seed")[metric]
                right = frame[frame.method == baseline].set_index("seed")[metric]
                paired = pd.concat([left.rename("head"), right.rename("baseline")], axis=1).dropna()
                diff = paired["head"].to_numpy() - paired["baseline"].to_numpy()
                samples = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(3000)])
                rows.append(
                    {
                        "head": head,
                        "comparison": f"{head} vs {baseline}",
                        "metric": metric,
                        "paired_mean_difference": float(diff.mean()),
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "wins": int((diff < 0).sum()),
                        "losses": int((diff > 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def l2_blend_ablation(data: base.PreparedData, seeds: list[int] | None = None) -> pd.DataFrame:
    """Separate the L2 stack, high-rank SoftImpute candidate, and validation blend."""
    rows: list[dict[str, object]] = []
    for seed in (seeds or SEEDS):
        test = base.make_random_mask(data, 0.20, seed)
        train = data.observed_mask & ~test
        rng = np.random.default_rng(seed + 4401)
        eligible = train.copy()
        eligible[:, train.sum(axis=0) < 24] = False
        validation = eligible & (rng.random(eligible.shape) < 0.16)
        if validation.sum() < 300:
            validation = eligible & (rng.random(eligible.shape) < 0.22)
        fit_mask = train & ~validation

        _, fit_candidates, _ = legacy._safe_stack_candidates(data, fit_mask, seed + 11, use_evidence=False)
        groups = np.array([base.group_id(feature) for feature in data.features])
        prior = np.array([0.10, 0.42, 0.24, 0.24])
        global_w = legacy.convex_weights(
            fit_candidates[:, validation].T,
            data.model_values[validation],
            prior=prior,
            ridge=0.07,
        )
        weights: dict[str, np.ndarray] = {}
        for group in sorted(set(groups)):
            mask = validation & (groups == group)[None, :]
            weights[group] = legacy.convex_weights(
                fit_candidates[:, mask].T,
                data.model_values[mask],
                prior=global_w,
                ridge=0.11,
            )

        _, select_candidates, _ = legacy._safe_stack_candidates(data, fit_mask, seed + 31, use_evidence=False)
        stack_fit = np.zeros_like(data.model_values)
        for j, group in enumerate(groups):
            stack_fit[:, j] = select_candidates[:, :, j].T @ weights[group]
        soft_fit = legacy.softimpute(data, fit_mask, rank=30, shrink=0.18, epochs=40)

        best_alpha = 1.0
        best_mse = np.inf
        for alpha in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]:
            candidate = alpha * stack_fit + (1.0 - alpha) * soft_fit
            candidate = legacy.exact_feasibility_projection(data, candidate)
            mse = float(np.mean((candidate[validation] - data.model_values[validation]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        _, candidates, _ = legacy._safe_stack_candidates(data, train, seed + 31, use_evidence=False)
        stack_pred = np.zeros_like(data.model_values)
        for j, group in enumerate(groups):
            stack_pred[:, j] = candidates[:, :, j].T @ weights[group]
        soft_pred = legacy.softimpute(data, train, rank=30, shrink=0.18, epochs=40)
        variants = [
            ("L2 pure stack", stack_pred, 1.0),
            ("High-rank SoftImpute alone", soft_pred, 0.0),
            ("Validation-selected L2 blend", best_alpha * stack_pred + (1.0 - best_alpha) * soft_pred, best_alpha),
        ]
        for variant, raw_pred, alpha in variants:
            pred = legacy.exact_feasibility_projection(data, raw_pred)
            result = base.evaluate(pred, data.model_values, test, None)
            result["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
            result.update(legacy.constraint_metrics(data, pred))
            result.update(
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_alpha_stack": float(alpha),
                    "validation_cells": int(validation.sum()),
                    "validation_mse_selected": best_mse,
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    city, codebook = base.read_inputs()
    data = base.prepare_features(city, codebook)
    existing = pd.read_csv(TABLE_DIR / "accuracy_feasibility_by_seed.csv", encoding="utf-8-sig")
    existing = existing[
        (existing.scenario == "random_20pct")
        & existing.method.isin(["Median+Proj", "SoftImpute+Proj", "Low-rank SVD+Proj", "PCCM+Proj"])
    ].copy()
    existing["method"] = existing["method"].replace({"PCCM+Proj": "PCCM-L1+Proj"})
    rows = [existing]
    l2_rows = []
    for seed in SEEDS:
        test = base.make_random_mask(data, 0.20, seed)
        train = data.observed_mask & ~test
        pred, _ = legacy.pccm_l2_predict(data, train, seed=seed, use_constraints=True)
        result = base.evaluate(pred, data.model_values, test, None)
        result["percent_mae_points"] = base.percent_mae_pp(data, pred, test)
        result.update(legacy.constraint_metrics(data, pred))
        result.update({"scenario": "random_20pct", "method": "PCCM-L2+Proj", "seed": seed})
        l2_rows.append(result)
    rows.append(pd.DataFrame(l2_rows))
    combined = pd.concat(rows, ignore_index=True, sort=False)
    summary = aggregate(combined, ["scenario", "method"])
    combined.to_csv(TABLE_DIR / "dual_head_random_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "dual_head_random_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap(combined).to_csv(TABLE_DIR / "dual_head_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    l2_ablation = l2_blend_ablation(data, SEEDS[:5])
    l2_ablation_summary = aggregate(l2_ablation, ["variant"])
    l2_ablation.to_csv(TABLE_DIR / "l2_blend_ablation_by_seed.csv", index=False, encoding="utf-8-sig")
    l2_ablation_summary.to_csv(TABLE_DIR / "l2_blend_ablation_summary.csv", index=False, encoding="utf-8-sig")
    print(summary[["method", "mae_norm", "mae_norm_std", "rmse_norm", "rmse_norm_std"]].sort_values("rmse_norm").to_string(index=False))
    print(l2_ablation_summary[["variant", "mae_norm", "rmse_norm", "selected_alpha_stack", "seeds"]].sort_values("rmse_norm").to_string(index=False))


if __name__ == "__main__":
    main()
