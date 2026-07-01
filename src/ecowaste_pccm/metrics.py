from __future__ import annotations

import numpy as np


def masked_errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    feature_scale: np.ndarray,
) -> dict[str, float]:
    if not mask.any():
        return {
            "normalized_mae": 0.0,
            "normalized_rmse": 0.0,
            "original_mae": 0.0,
            "original_rmse": 0.0,
            "evaluated_cells": 0,
        }
    err = prediction[mask] - truth[mask]
    scale = np.broadcast_to(feature_scale, truth.shape)[mask]
    norm_err = err / np.maximum(scale, 1e-9)
    return {
        "normalized_mae": float(np.mean(np.abs(norm_err))),
        "normalized_rmse": float(np.sqrt(np.mean(norm_err * norm_err))),
        "original_mae": float(np.mean(np.abs(err))),
        "original_rmse": float(np.sqrt(np.mean(err * err))),
        "evaluated_cells": int(mask.sum()),
    }
