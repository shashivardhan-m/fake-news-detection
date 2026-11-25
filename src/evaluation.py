import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


class ModelEvaluator:
    """
    Comprehensive model evaluator:
    - Standard metrics: Accuracy, Precision, Recall, F1
    - Robust classification metrics: Macro F1, Weighted F1, PR-AUC, ROC-AUC
    - Calibration assessment: Brier Score, Expected Calibration Error (ECE)
    - Detailed Confusion Matrix breakdown (TP, FP, TN, FN)
    - Diagnostic visualizations: Confusion Matrix, PR Curve, ROC Curve, Calibration Curve
    """
    def __init__(self, y_true, y_pred, y_proba=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None

    def get_metrics(self):
        metrics = {
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            'precision': float(precision_score(self.y_true, self.y_pred, zero_division=0)),
            'recall': float(recall_score(self.y_true, self.y_pred, zero_division=0)),
            'f1': float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            'macro_f1': float(f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)),
        }

        cm = self.get_confusion_matrix()
        if cm.shape == (2, 2):
            metrics['true_negatives'] = int(cm[0, 0])
            metrics['false_positives'] = int(cm[0, 1])
            metrics['false_negatives'] = int(cm[1, 0])
            metrics['true_positives'] = int(cm[1, 1])

        if self.y_proba is not None and len(np.unique(self.y_true)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(self.y_true, self.y_proba))
            except Exception:
                pass
            try:
                metrics['pr_auc'] = float(average_precision_score(self.y_true, self.y_proba))
            except Exception:
                pass
            try:
                metrics['brier_score'] = float(brier_score_loss(self.y_true, self.y_proba))
            except Exception:
                pass

            metrics['ece'] = float(self.calculate_ece())

        return metrics

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self):
        return classification_report(self.y_true, self.y_pred, zero_division=0, output_dict=False)

    def calculate_ece(self, n_bins=10):
        """Calculates Expected Calibration Error (ECE)."""
        if self.y_proba is None:
            return 0.0

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        n_samples = len(self.y_true)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (self.y_proba > bin_lower) & (self.y_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(self.y_true[in_bin])
                avg_confidence_in_bin = np.mean(self.y_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def plot_confusion_matrix(self, save_path=None, title='Confusion Matrix'):
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'],
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_calibration_curve(self, save_path=None, n_bins=10, title='Probability Calibration Curve'):
        if self.y_proba is None:
            return

        prob_true, prob_pred = calibration_curve(self.y_true, self.y_proba, n_bins=n_bins)
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives (True Frequency)')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def cross_validate_metrics(model, X, y, cv=5, n_repeats=1, random_state=42):
        """
        Runs Stratified K-Fold or Repeated Stratified K-Fold CV,
        reporting both mean and standard deviation for all primary metrics.
        """
        if n_repeats > 1:
            splitter = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_state)
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        accs, precs, recs, f1s, macro_f1s, roc_aucs, pr_aucs = [], [], [], [], [], [], []

        for train_idx, val_idx in splitter.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            macro_f1s.append(f1_score(y_val, y_pred, average='macro', zero_division=0))

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_val)[:, 1]
                try:
                    roc_aucs.append(roc_auc_score(y_val, proba))
                except Exception:
                    pass
                try:
                    pr_aucs.append(average_precision_score(y_val, proba))
                except Exception:
                    pass

        results = {
            'cv_accuracy_mean': float(np.mean(accs)),
            'cv_accuracy_std': float(np.std(accs)),
            'cv_precision_mean': float(np.mean(precs)),
            'cv_precision_std': float(np.std(precs)),
            'cv_recall_mean': float(np.mean(recs)),
            'cv_recall_std': float(np.std(recs)),
            'cv_f1_mean': float(np.mean(f1s)),
            'cv_f1_std': float(np.std(f1s)),
            'cv_macro_f1_mean': float(np.mean(macro_f1s)),
            'cv_macro_f1_std': float(np.std(macro_f1s)),
        }

        if roc_aucs:
            results['cv_roc_auc_mean'] = float(np.mean(roc_aucs))
            results['cv_roc_auc_std'] = float(np.std(roc_aucs))
        if pr_aucs:
            results['cv_pr_auc_mean'] = float(np.mean(pr_aucs))
            results['cv_pr_auc_std'] = float(np.std(pr_aucs))

        return results
