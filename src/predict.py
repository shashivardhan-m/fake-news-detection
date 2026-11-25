import json
import os
import re

from explainability import explain_prediction_breakdown, KNOWN_PUBLISHER_CUES
from feature_extraction import FeatureExtractor
from linguistic_features import analyze_text, get_risk_flags
from models import FakeNewsModel, MODEL_CONFIGS
from preprocessing import TextPreprocessor

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
    'transformer': 'transformer',
}

MIN_WORDS_THRESHOLD = 15
FACT_CHECK_DISCLAIMER = (
    "DISCLAIMER: This system classifies stylistic, linguistic, and lexical patterns learned "
    "from training news datasets. It does NOT verify real-world factual claims against external "
    "evidence, primary sources, or knowledge bases. High confidence reflects stylistic alignment, "
    "not guaranteed truth or falsehood."
)


def _project_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    return project_dir, models_dir


def check_input_sufficiency(text):
    """
    Checks if input text contains enough tokens for reliable classification.
    Returns (is_sufficient, word_count, warning_message).
    """
    words = re.findall(r'\b\w+\b', text)
    count = len(words)
    if count == 0:
        return False, 0, "Input is empty. Please enter news headline and article text."
    if count < MIN_WORDS_THRESHOLD:
        return (
            False,
            count,
            f"Input contains only {count} words (minimum recommended: {MIN_WORDS_THRESHOLD}). "
            "Text classifiers trained on article bodies may produce spurious predictions on short snippets.",
        )
    return True, count, None


def check_publisher_cues_in_text(text):
    """Detects whether text contains explicit news agency markers."""
    lower = text.lower()
    found = [cue for cue in KNOWN_PUBLISHER_CUES if re.search(rf'\b{re.escape(cue)}\b', lower)]
    return found


def predict_news(text, model_type='ensemble', explain=False, strip_cues=True):
    """
    Predicts whether an article is Real or Fake with:
    - Input sufficiency guard
    - Publisher cue detection
    - Calibrated confidence estimate
    - Factual verification disclaimer
    - Explainability breakdown (words supporting Real vs Fake)
    """
    project_dir, models_dir = _project_paths()
    model_type = MODEL_ALIASES.get(model_type, model_type)

    is_sufficient, word_count, length_warning = check_input_sufficiency(text)
    publisher_cues = check_publisher_cues_in_text(text)
    linguistic_stats = analyze_text(text)
    risk_flags = get_risk_flags(linguistic_stats)

    preprocessor = TextPreprocessor(
        preserve_negation=True,
        preserve_numbers=True,
        strip_publisher_cues=strip_cues,
    )
    cleaned_text = preprocessor.preprocess(text)

    # 1. PyTorch Deep Learning Models
    if model_type in ('lstm', 'transformer'):
        from deep_learning import LSTMModel, TransformerModel
        cls = LSTMModel if model_type == 'lstm' else TransformerModel
        prefix = 'lstm' if model_type == 'lstm' else 'transformer'

        model_path = os.path.join(models_dir, f'{prefix}_model.pt')
        tok_path = os.path.join(models_dir, f'{prefix}_tokenizer.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Train it first using train.py.")

        dl_model = cls()
        dl_model.load(model_path, tok_path)

        prediction = dl_model.predict([cleaned_text])[0]
        probability = dl_model.predict_proba([cleaned_text])[0]

        conf_estimate = float(max(probability))

        return {
            'prediction': int(prediction),
            'label': 'Fake' if prediction == 1 else 'Real',
            'model_confidence_estimate': round(conf_estimate, 4),
            'probability': {'real': round(float(probability[0]), 4), 'fake': round(float(probability[1]), 4)},
            'input_sufficient': is_sufficient,
            'word_count': word_count,
            'length_warning': length_warning,
            'publisher_cues_found': publisher_cues,
            'linguistic_stats': linguistic_stats,
            'risk_flags': risk_flags,
            'explanations': None,
            'disclaimer': FACT_CHECK_DISCLAIMER,
        }

    # 2. Scikit-learn Classical Models
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}. Train models first.")

    extractor = FeatureExtractor()
    extractor.load(vectorizer_path)
    features = extractor.transform([cleaned_text])

    model = FakeNewsModel(model_type=model_type)
    model_file = os.path.join(models_dir, FakeNewsModel.get_filename(model_type))
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Train models first.")
    model.load(model_file)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    conf_estimate = float(max(probability))

    explanations = None
    if explain:
        inner_model = model.model
        if hasattr(inner_model, 'calibrated_classifiers_'):
            inner_model = inner_model.calibrated_classifiers_[0].estimator
        feature_names = extractor.get_feature_names()
        explanations = explain_prediction_breakdown(inner_model, features, feature_names)

    return {
        'prediction': int(prediction),
        'label': 'Fake' if prediction == 1 else 'Real',
        'model_confidence_estimate': round(conf_estimate, 4),
        'probability': {'real': round(float(probability[0]), 4), 'fake': round(float(probability[1]), 4)},
        'input_sufficient': is_sufficient,
        'word_count': word_count,
        'length_warning': length_warning,
        'publisher_cues_found': publisher_cues,
        'linguistic_stats': linguistic_stats,
        'risk_flags': risk_flags,
        'explanations': explanations,
        'disclaimer': FACT_CHECK_DISCLAIMER,
    }


def predict_all_models(text, strip_cues=True):
    project_dir, models_dir = _project_paths()
    results = {}

    for model_type in list(MODEL_CONFIGS.keys()) + ['ensemble']:
        path = os.path.join(models_dir, FakeNewsModel.get_filename(model_type))
        if os.path.exists(path):
            name = FakeNewsModel.get_display_name(model_type)
            results[name] = predict_news(text, model_type, strip_cues=strip_cues)

    for dl_type, display_name in [('lstm', 'Bidirectional LSTM'), ('transformer', 'Self-Attention Transformer')]:
        pt_path = os.path.join(models_dir, f'{dl_type}_model.pt')
        if os.path.exists(pt_path):
            results[display_name] = predict_news(text, dl_type, strip_cues=strip_cues)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <news text> [--model ensemble] [--explain] [--no-strip-cues]")
        sys.exit(1)

    args = sys.argv[1:]
    explain = '--explain' in args
    strip_cues = '--no-strip-cues' not in args

    model_type = 'ensemble'
    if '--model' in args:
        idx = args.index('--model')
        model_type = args[idx + 1]
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    clean_args = [a for a in args if a not in ('--explain', '--no-strip-cues')]
    text = ' '.join(clean_args)

    result = predict_news(text, model_type=model_type, explain=explain, strip_cues=strip_cues)

    print(f"\nPrediction: {result['label']}")
    print(f"Model Confidence Estimate: {result['model_confidence_estimate']:.1%}")
    print(f"Probability Distribution: Real: {result['probability']['real']:.2%} | Fake: {result['probability']['fake']:.2%}")

    if result['length_warning']:
        print(f"\n[!] Input Notice: {result['length_warning']}")

    if result['publisher_cues_found']:
        print(f"\n[!] Source Leakage Warning: Found publisher markers: {result['publisher_cues_found']}")

    if result['risk_flags']:
        print("\nStylistic Observations (Heuristics only):")
        for flag in result['risk_flags']:
            print(f"  - {flag}")

    if result['explanations']:
        print("\nTop Contributing Features (Evidence Alignment):")
        if result['explanations']['fake_evidence']:
            print("  Supporting Fake:")
            for item in result['explanations']['fake_evidence'][:5]:
                print(f"    + {item['feature']}: {item['contribution']:+.4f}")
        if result['explanations']['real_evidence']:
            print("  Supporting Real:")
            for item in result['explanations']['real_evidence'][:5]:
                print(f"    - {item['feature']}: {item['contribution']:+.4f}")

    print(f"\n{result['disclaimer']}")
