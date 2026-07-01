from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelOutput:
    prediction_norm: np.ndarray
    uncertainty_norm: np.ndarray
    weights: dict[str, float]
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def make_scaler(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(train_mask, values, np.nan)
    mean = np.nanmedian(masked, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    q75 = np.nanpercentile(masked, 75, axis=0)
    q25 = np.nanpercentile(masked, 25, axis=0)
    scale = q75 - q25
    q90 = np.nanpercentile(masked, 90, axis=0)
    q10 = np.nanpercentile(masked, 10, axis=0)
    wide_scale = q90 - q10
    fallback = np.nanstd(masked, axis=0)
    scale = np.where(
        np.isfinite(wide_scale) & (wide_scale > 1e-9),
        np.maximum(scale, 0.25 * wide_scale),
        scale,
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    return mean, scale


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def denormalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values * scale + mean


def global_median(values_norm: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    masked = np.where(train_mask, values_norm, np.nan)
    med = np.nanmedian(masked, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    pred = np.tile(med, (values_norm.shape[0], 1))
    pred[train_mask] = values_norm[train_mask]
    return pred


def group_median(values_norm: np.ndarray, train_mask: np.ndarray, keys: list[tuple[str, str]]) -> np.ndarray:
    pred = global_median(values_norm, train_mask)
    keys_arr = np.array([f"{a}|{b}" for a, b in keys])
    for key in np.unique(keys_arr):
        rows = keys_arr == key
        if rows.sum() < 2:
            continue
        group_mask = train_mask[rows]
        group_values = values_norm[rows]
        has_observed = group_mask.any(axis=0)
        if not has_observed.any():
            continue
        med = np.full(values_norm.shape[1], np.nan)
        med[has_observed] = np.nanmedian(
            np.where(group_mask[:, has_observed], group_values[:, has_observed], np.nan),
            axis=0,
        )
        use = np.isfinite(med)
        pred[np.ix_(rows, use)] = med[use]
    pred[train_mask] = values_norm[train_mask]
    return pred


def low_rank_svd(
    values_norm: np.ndarray,
    train_mask: np.ndarray,
    rank: int = 6,
    iterations: int = 25,
) -> np.ndarray:
    filled = global_median(values_norm, train_mask)
    rank = max(1, min(rank, min(values_norm.shape) - 1))
    for _ in range(iterations):
        centered = filled - filled.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        approx += filled.mean(axis=0, keepdims=True)
        filled[~train_mask] = approx[~train_mask]
        filled[train_mask] = values_norm[train_mask]
    return filled


def knn_expert(values_norm: np.ndarray, train_mask: np.ndarray, k: int = 8) -> np.ndarray:
    base = global_median(values_norm, train_mask)
    n_rows, n_cols = values_norm.shape
    pred = base.copy()
    distances = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(n_rows):
        diff = base - base[i]
        distances[i] = np.sqrt(np.nanmean(diff * diff, axis=1))
        distances[i, i] = np.inf
    order = np.argsort(distances, axis=1)
    for i in range(n_rows):
        neighbors = order[i]
        for j in range(n_cols):
            obs_neighbors = neighbors[train_mask[neighbors, j]][:k]
            if len(obs_neighbors):
                pred[i, j] = float(np.mean(values_norm[obs_neighbors, j]))
    pred[train_mask] = values_norm[train_mask]
    return pred


def learn_simplex_weights(
    expert_predictions: dict[str, np.ndarray],
    values_norm: np.ndarray,
    validation_mask: np.ndarray,
) -> dict[str, float]:
    names = list(expert_predictions)
    if not validation_mask.any():
        return {name: 1.0 / len(names) for name in names}

    y = values_norm[validation_mask]
    mae_scores = []
    rmse_scores = []
    for name in names:
        err = expert_predictions[name][validation_mask] - y
        mae_scores.append(float(np.mean(np.abs(err))))
        rmse_scores.append(float(np.sqrt(np.mean(err * err))))

    mae_scores = np.array(mae_scores)
    rmse_scores = np.array(rmse_scores)
    l1 = 1.0 / np.maximum(mae_scores, 1e-9)
    l2 = 1.0 / np.maximum(rmse_scores, 1e-9)
    weights = 0.5 * (l1 / l1.sum()) + 0.5 * (l2 / l2.sum())
    return {name: float(weight) for name, weight in zip(names, weights)}


def fit_pccm(
    values: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    keys: list[tuple[str, str]],
    svd_rank: int,
    svd_iterations: int,
    knn_neighbors: int,
) -> ModelOutput:
    feature_mean, feature_scale = make_scaler(values, fit_mask)
    values_norm = normalize(values, feature_mean, feature_scale)

    experts = {
        "global_median": global_median(values_norm, fit_mask),
        "region_income_median": group_median(values_norm, fit_mask, keys),
        "low_rank": low_rank_svd(values_norm, fit_mask, svd_rank, svd_iterations),
        "city_graph_knn": knn_expert(values_norm, fit_mask, knn_neighbors),
    }
    weights = learn_simplex_weights(experts, values_norm, validation_mask)

    prediction = np.zeros_like(values_norm, dtype=float)
    stack = []
    for name, pred in experts.items():
        prediction += weights[name] * pred
        stack.append(pred)
    uncertainty = np.std(np.stack(stack, axis=0), axis=0)

    known = fit_mask | validation_mask
    prediction[known] = values_norm[known]
    uncertainty[known] = 0.0
    return ModelOutput(prediction, uncertainty, weights, feature_mean, feature_scale)
