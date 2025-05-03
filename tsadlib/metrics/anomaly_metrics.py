"""
=================================================
@Author: Zenon
@Date: 2025-05-02
@Description: Comprehensive anomaly detection metrics calculation module.
              Provides common metrics (precision/recall/F1/AUC), enhanced metrics
              (affiliation/VUS), and point adjustment functionality.
==================================================
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .threshold.basics import percentile_threshold, best_f1_threshold
from .vus.metrics import get_range_vus_roc
from .. import Metric


class AnomalyMetrics:
    """
    Unified interface for calculating anomaly detection performance metrics.
    
    Attributes:
        labels (np.ndarray): Ground truth labels (1: anomaly, 0: normal)
        scores (np.ndarray): Predicted anomaly scores
        pred_labels (np.ndarray): Binary predictions after thresholding
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray, threshold_way: str = 'best_f1', anomaly_rate: float = 1.,
                 train_scores: np.ndarray = None):
        """
        Initialize metrics calculator with ground truth and predictions.
        
        Args:
            labels: Ground truth binary labels
            scores: Predicted anomaly scores (higher values indicate more anomalous)
            threshold_way: Threshold selection method ('best_f1' or 'percentile')
            anomaly_rate: Expected anomaly rate for percentile thresholding
            train_scores: Predicted anomaly scores in training set.
        """
        self.scores = scores
        self.labels = labels
        if threshold_way == 'percentile':
            combined_scores = np.concatenate([train_scores, self.scores], axis=0)
            threshold = percentile_threshold(combined_scores, 100 - anomaly_rate)
        else:
            threshold = best_f1_threshold(self.scores, self.labels)

        self.pred_labels = (self.scores > threshold).astype(int)

    def common_metrics(self) -> Metric:
        """
        Calculate basic anomaly detection metrics.

        Returns:
            Dictionary containing:
            - Precision: TP / (TP + FP)
            - Recall: TP / (TP + FN) 
            - F1_score: Harmonic mean of precision and recall
            - ROC_AUC: Area under ROC curve
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, self.pred_labels, average='binary')

        return Metric(**{
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1,
            'ROC_AUC': roc_auc_score(self.labels, self.scores)
        })

    def enhanced_metrics(self) -> Metric:
        """
        Calculate advanced time-series specific metrics.
        
        Returns:
            Dictionary containing:
            - Affiliation_Precision: Precision considering event alignment
            - Affiliation_Recall: Recall considering event alignment
            - R_AUC_ROC: Range-based AUC for ROC curve
            - R_AUC_PR: Range-based AUC for PR curve
            - VUS_ROC: Volume under ROC surface
            - VUS_PR: Volume under PR surface
        """
        events_pred = convert_vector_to_events(self.pred_labels)
        events_gt = convert_vector_to_events(self.labels)
        Trange = (0, len(self.pred_labels))

        affiliation = pr_from_events(events_pred, events_gt, Trange)
        vus_results = get_range_vus_roc(self.scores, self.labels, 100)

        return Metric(**{
            'Affiliation_Precision': affiliation['Affiliation_Precision'],
            'Affiliation_Recall': affiliation['Affiliation_Recall'],
            'R_AUC_ROC': vus_results["R_AUC_ROC"],
            'R_AUC_PR': vus_results["R_AUC_PR"],
            'VUS_ROC': vus_results["VUS_ROC"],
            'VUS_PR': vus_results["VUS_PR"]
        })

    def point_adjustment(self) -> None:
        """
        Apply point adjustment strategy to predictions.
        Expands predicted anomalies to cover entire ground truth anomaly segments.
        Modifies pred_labels in-place.
        """
        anomaly_state = False
        for i in range(len(self.labels)):
            if self.labels[i] == 1 and self.pred_labels[i] == 1 and not anomaly_state:
                anomaly_state = True
                # Expand backwards
                for j in range(i, 0, -1):
                    if self.labels[j] == 0:
                        break
                    elif self.pred_labels[j] == 0:
                        self.pred_labels[j] = 1
                # Expand forwards
                for j in range(i, len(self.labels)):
                    if self.labels[j] == 0:
                        break
                    elif self.pred_labels[j] == 0:
                        self.pred_labels[j] = 1
            elif self.labels[i] == 0:
                anomaly_state = False
            if anomaly_state:
                self.pred_labels[i] = 1
