"""
Model evaluation utilities for FloodML
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
        """Evaluate binary classification performance."""
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
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "critical_success_index": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        }

    def evaluate_probabilistic_forecast(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate probabilistic forecast (Brier score, skill score, etc.)."""
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        climatology = np.mean(y_true)
        brier_score_ref = np.mean((climatology - y_true) ** 2)

        return {
            "brier_score": brier_score,
            "brier_skill_score": 1 - (brier_score / brier_score_ref)
            if brier_score_ref > 0 else 0
        }

    def calculate_optimal_threshold(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = 'f1'
    ) -> Tuple[float, float]:
        """Find optimal probability threshold for classification."""
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

    def generate_evaluation_report(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str = "FloodModel"
    ) -> Dict[str, Any]:
        """Generate a complete evaluation report for both binary and probabilistic metrics."""
        binary_metrics = self.evaluate_binary_classifier(y_true, y_pred_proba)
        prob_metrics = self.evaluate_probabilistic_forecast(y_true, y_pred_proba)
        f1_thr, f1_score_val = self.calculate_optimal_threshold(y_true, y_pred_proba, 'f1')

        return {
            "model_name": model_name,
            "binary_metrics": binary_metrics,
            "probabilistic_metrics": prob_metrics,
            "optimal_f1_threshold": f1_thr,
            "optimal_f1_score": f1_score_val
        }
