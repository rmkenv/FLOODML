"""
Model evaluation utilities for FloodML
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import structlog

logger = structlog.get_logger()

class ModelEvaluator:
    """
    Comprehensive model evaluation for flood prediction models
    
    Provides metrics specifically relevant for rare event prediction
    and time series validation.
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_binary_classifier(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate binary classification performance
        """
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        }
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'probability_of_detection': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_alarm_rate': fp / (tp + fp) if (tp + fp) > 0 else 0,
            'critical_success_index': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        })
        return metrics
    
    def evaluate_probabilistic_forecast(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate probabilistic forecast quality
        """
        metrics = {}
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        metrics['brier_score'] = brier_score
        climatology = np.mean(y_true)
        brier_score_ref = np.mean((climatology - y_true) ** 2)
        metrics['brier_skill_score'] = 1 - (brier_score / brier_score_ref) if brier_score_ref > 0 else 0
        
        # Reliability and resolution (simplified)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        reliability = 0
        resolution = 0
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_freq = np.sum(mask) / len(y_true)
                bin_prob = bin_centers[i]
                bin_outcome = np.mean(y_true[mask])
                reliability += bin_freq * (bin_prob - bin_outcome) ** 2
                resolution += bin_freq * (bin_outcome - climatology) ** 2
        
        metrics['reliability'] = reliability
        metrics['resolution'] = resolution
        return metrics
    
    def calculate_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal probability threshold for classification
        """
        thresholds = np.linspace(0, 1, 101)
        scores = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'csi':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores.append(score)
        optimal_idx = np.argmax(scores)
        return thresholds[optimal_idx], scores[optimal_idx]
    
    def generate_evaluation_report(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        model_name: str = "FloodModel"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        report = {
            'model_name': model_name,
            'dataset_stats': {
                'total_samples': len(y_true),
                'flood_events': int(np.sum(y_true)),
                'flood_rate': float(np.mean(y_true)),
                'no_flood_events': int(len(y_true) - np.sum(y_true))
            }
        }
        binary_metrics = self.evaluate_binary_classifier(y_true, y_pred_proba)
        report['binary_metrics'] = binary_metrics
        
        prob_metrics = self.evaluate_probabilistic_forecast(y_true, y_pred_proba)
        report['probabilistic_metrics'] = prob_metrics
        
        optimal_f1_thresh, optimal_f1_score = self.calculate_optimal_threshold(y_true, y_pred_proba, 'f1')
        optimal_csi_thresh, optimal_csi_score = self.calculate_optimal_threshold(y_true, y_pred_proba, 'csi')
        
        report['threshold_analysis'] = {
            'optimal_f1_threshold': optimal_f1_thresh,
            'optimal_f1_score': optimal_f1_score,
            'optimal_csi_threshold': optimal_csi_thresh,
            'optimal_csi_score': optimal_csi_score
        }
        return report
    
    def print_evaluation_summary(self, report: Dict[str, Any]):
        print(f"📊 Evaluation Report: {report['model_name']}")
        print("=" * 50)
        stats = report['dataset_stats']
        print(f"📈 Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']:,}")
        print(f"   Flood events: {stats['flood_events']:,} ({stats['flood_rate']:.1%})")
        print(f"   No-flood events: {stats['no_flood_events']:,}")
        metrics = report['binary_metrics']
        print(f"\n🎯 Performance Metrics:")
        print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall (POD): {metrics['recall']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   Critical Success Index: {metrics['critical_success_index']:.3f}")
        thresh = report['threshold_analysis']
        print(f"\n🔧 Optimal Thresholds:")
        print(f"   Best F1: {thresh['optimal_f1_score']:.3f} @ {thresh['optimal_f1_threshold']:.2f}")
        print(f"   Best CSI: {thresh['optimal_csi_score']:.3f} @ {thresh['optimal_csi_threshold']:.2f}")
