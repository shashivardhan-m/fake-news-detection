from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import FakeNewsModel

def predict_news(text):
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.preprocess(text)
    
    extractor = FeatureExtractor()
    extractor.load('models/tfidf_vectorizer.joblib')
    features = extractor.transform([cleaned_text])
    
    model = FakeNewsModel()
    model.load('models/fake_news_model.joblib')
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        prediction, probability = predict_news(text)
        label = "Fake" if prediction == 1 else "Real"
        print(f"Prediction: {label}")
        print(f"Probability (Real): {probability[0]:.4f}")
        print(f"Probability (Fake): {probability[1]:.4f}")
    else:
        print("Usage: python predict.py <news text>")

