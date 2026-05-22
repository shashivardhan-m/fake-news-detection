from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.is_fitted = False
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)
    
    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True

