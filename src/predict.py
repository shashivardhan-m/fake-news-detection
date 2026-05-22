import os
from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import FakeNewsModel

def predict_news(text, model_type='logistic_regression'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.preprocess(text)
    
    extractor = FeatureExtractor()
    extractor.load(os.path.join(project_dir, 'models', 'tfidf_vectorizer.joblib'))
    features = extractor.transform([cleaned_text])
    
    model = FakeNewsModel(model_type=model_type)
    if model_type == 'logistic_regression':
        model.load(os.path.join(project_dir, 'models', 'logistic_regression.joblib'))
    else:
        model.load(os.path.join(project_dir, 'models', 'random_forest.joblib'))
    
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

