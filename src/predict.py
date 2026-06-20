import os
import json

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from linguistic_features import analyze_text, get_risk_flags


MODEL_ALIASES = {
    'lr': 'logistic_regression',
    'logistic': 'logistic_regression',
    'logistic_regression': 'logistic_regression',
    'rf': 'random_forest',
    'random_forest': 'random_forest',
    'gb': 'gradient_boosting',
    'gradient_boosting': 'gradient_boosting',
    'svm': 'svm',
    'ensemble': 'ensemble',
    'lstm': 'lstm',
}


def _project_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    return project_dir, models_dir


def predict_news(text, model_type='ensemble', explain=False):
    project_dir, models_dir = _project_paths()
    model_type = MODEL_ALIASES.get(model_type, model_type)

    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.preprocess(text)
    linguistic_stats = analyze_text(text)
    risk_flags = get_risk_flags(linguistic_stats)

    if model_type == 'lstm':
        from deep_learning import LSTMModel
        lstm = LSTMModel()
        lstm.load(
            os.path.join(models_dir, 'lstm_model.keras'),
            os.path.join(models_dir, 'lstm_tokenizer.joblib'),
        )
        prediction = lstm.predict([cleaned_text])[0]
        probability = lstm.predict_proba([cleaned_text])[0]
        return {
            'prediction': int(prediction),
            'label': 'Fake' if prediction == 1 else 'Real',
            'probability': {'real': float(probability[0]), 'fake': float(probability[1])},
            'linguistic_stats': linguistic_stats,
            'risk_flags': risk_flags,
            'explanations': None,
        }

    extractor = FeatureExtractor()
    extractor.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    features = extractor.transform([cleaned_text])

    model = FakeNewsModel(model_type=model_type)
    model.load(os.path.join(models_dir, FakeNewsModel.get_filename(model_type)))

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    explanations = None
    if explain:
        from explainability import explain_with_shap
        inner_model = model.model
        if hasattr(inner_model, 'calibrated_classifiers_'):
            inner_model = inner_model.calibrated_classifiers_[0].estimator
        feature_names = extractor.get_feature_names()
        explanations = explain_with_shap(inner_model, features, feature_names)

    return {
        'prediction': int(prediction),
        'label': 'Fake' if prediction == 1 else 'Real',
        'probability': {'real': float(probability[0]), 'fake': float(probability[1])},
        'linguistic_stats': linguistic_stats,
        'risk_flags': risk_flags,
        'explanations': explanations,
    }


def predict_all_models(text):
    project_dir, models_dir = _project_paths()
    results = {}

    for model_type in list(MODEL_CONFIGS.keys()) + ['ensemble']:
        path = os.path.join(models_dir, FakeNewsModel.get_filename(model_type))
        if os.path.exists(path):
            results[FakeNewsModel.get_display_name(model_type)] = predict_news(text, model_type)

    lstm_path = os.path.join(models_dir, 'lstm_model.keras')
    if os.path.exists(lstm_path):
        results['Bidirectional LSTM'] = predict_news(text, 'lstm')

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <news text> [--model ensemble] [--explain]")
        sys.exit(1)

    args = sys.argv[1:]
    explain = '--explain' in args
    model_type = 'ensemble'
    if '--model' in args:
        idx = args.index('--model')
        model_type = args[idx + 1]
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
    args = [a for a in args if a != '--explain']

    text = ' '.join(args)
    result = predict_news(text, model_type=model_type, explain=explain)

    print(f"Prediction: {result['label']}")
    print(f"Real: {result['probability']['real']:.2%} | Fake: {result['probability']['fake']:.2%}")

    if result['risk_flags']:
        print("\nLinguistic risk flags:")
        for flag in result['risk_flags']:
            print(f"  - {flag}")

    if result['explanations']:
        print("\nTop contributing features:")
        for item in result['explanations']:
            print(f"  {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
