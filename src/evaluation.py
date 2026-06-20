from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def get_metrics(self):
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        if self.y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
        return metrics

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self):
        return classification_report(self.y_true, self.y_pred, zero_division=0)

    def plot_confusion_matrix(self, save_path=None):
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def cross_validate_metrics(model, X, y, cv=5):
        from sklearn.model_selection import cross_val_score
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        results = {}
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            results[f'cv_{metric}_mean'] = float(np.mean(scores))
            results[f'cv_{metric}_std'] = float(np.std(scores))
        return results
