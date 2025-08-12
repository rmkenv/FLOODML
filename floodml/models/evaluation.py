"""
Model evaluation utilities for FLOODML
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

class ModelEvaluator:
    """Provides evaluation metrics for binary flood prediction models."""

    def evaluate_binary_classifier(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        return {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "critical_success_index": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        }

    def calculate_optimal_threshold(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = 'f1'
    ) -> Tuple[float, float]:
        thresholds = np.linspace(0, 1, 101)
        best_thr, best_score = 0.5, -np.inf
        for thr in thresholds:
            y_bin = (y_pred_proba >= thr).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_bin, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_bin, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_bin, zero_division=0)
            else:
                continue
            if score > best_score:
                best_thr, best_score = thr, score
        return best_thr, best_score
