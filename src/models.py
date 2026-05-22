from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

class FakeNewsModel:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)

