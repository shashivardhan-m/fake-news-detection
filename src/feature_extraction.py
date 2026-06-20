from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureExtractor:
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
        )
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

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
