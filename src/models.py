from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'kwargs': {'max_iter': 2000, 'C': 1.0, 'class_weight': 'balanced'},
        'filename': 'logistic_regression.joblib',
        'display_name': 'Logistic Regression',
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'kwargs': {'n_estimators': 200, 'max_depth': 30, 'class_weight': 'balanced', 'n_jobs': -1},
        'filename': 'random_forest.joblib',
        'display_name': 'Random Forest',
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'kwargs': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5},
        'filename': 'gradient_boosting.joblib',
        'display_name': 'Gradient Boosting',
    },
    'svm': {
        'class': LinearSVC,
        'kwargs': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 3000},
        'filename': 'svm.joblib',
        'display_name': 'Linear SVM',
        'calibrated': True,
    },
}


class FakeNewsModel:
    def __init__(self, model_type='logistic_regression'):
        if model_type not in MODEL_CONFIGS and model_type != 'ensemble':
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_type = model_type
        self.model = None
        self.ensemble_members = []

        if model_type == 'ensemble':
            return

        config = MODEL_CONFIGS[model_type]
        base_model = config['class'](**config['kwargs'])

        if config.get('calibrated'):
            self.model = CalibratedClassifierCV(base_model, cv=3)
        else:
            self.model = base_model

    def build_ensemble(self, models):
        estimators = [(name, model.model) for name, model in models]
        self.ensemble_members = [name for name, _ in estimators]
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1,
        )
        self.model_type = 'ensemble'

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
            import numpy as np
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
        raise AttributeError(f"Model {self.model_type} does not support probability estimates")

    def save(self, filepath):
        joblib.dump({'model': self.model, 'model_type': self.model_type}, filepath)

    def load(self, filepath):
        payload = joblib.load(filepath)
        if isinstance(payload, dict):
            self.model = payload['model']
            self.model_type = payload.get('model_type', self.model_type)
        else:
            self.model = payload

    @staticmethod
    def get_filename(model_type):
        if model_type == 'ensemble':
            return 'ensemble.joblib'
        return MODEL_CONFIGS[model_type]['filename']

    @staticmethod
    def get_display_name(model_type):
        if model_type == 'ensemble':
            return 'Ensemble (Voting)'
        return MODEL_CONFIGS[model_type]['display_name']
