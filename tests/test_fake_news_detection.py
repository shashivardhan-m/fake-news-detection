import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from deep_learning import LSTMModel, TransformerModel
from evaluation import ModelEvaluator
from explainability import explain_prediction_breakdown
from feature_extraction import FeatureExtractor
from linguistic_features import analyze_text, evaluate_heuristic_risk_flags, get_risk_flags
from models import FakeNewsModel
from predict import check_input_sufficiency, check_publisher_cues_in_text, predict_news
from preprocessing import TextPreprocessor, strip_publisher_artifacts


# ── 1. Empty Input Tests ──────────────────────────────────────────────────────

def test_preprocessing_empty_and_whitespace():
    preprocessor = TextPreprocessor()
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess("    ") == ""
    assert preprocessor.preprocess(None) == ""
    assert preprocessor.preprocess(12345) == ""


def test_analyze_text_empty():
    stats = analyze_text("")
    assert stats['word_count'] == 0
    assert stats['sentence_count'] == 0
    assert stats['sensational_score'] == 0.0
    flags = get_risk_flags(stats)
    assert flags == []


def test_input_sufficiency_empty():
    suff, count, msg = check_input_sufficiency("")
    assert not suff
    assert count == 0
    assert "empty" in msg.lower()


# ── 2. Very Short News Articles Tests ─────────────────────────────────────────

def test_short_news_articles_flagged():
    short_text = "Breaking news today."
    suff, count, msg = check_input_sufficiency(short_text)
    assert not suff
    assert count == 3
    assert "minimum recommended" in msg


def test_predict_news_short_input_warning():
    res = predict_news("Breaking news alert today.", model_type='logistic_regression')
    assert not res['input_sufficient']
    assert res['length_warning'] is not None
    assert "words" in res['length_warning']


# ── 3. Negation Preservation Tests ───────────────────────────────────────────

def test_negation_word_preservation():
    p_neg = TextPreprocessor(preserve_negation=True, preserve_numbers=False)
    p_no_neg = TextPreprocessor(preserve_negation=False, preserve_numbers=False)

    s1 = "The minister did not resign."
    cleaned_neg = p_neg.preprocess(s1)
    cleaned_no_neg = p_no_neg.preprocess(s1)

    assert "not" in cleaned_neg
    assert "not" not in cleaned_no_neg
    assert cleaned_neg != cleaned_no_neg


def test_contraction_expansion_with_negation():
    p = TextPreprocessor(preserve_negation=True)
    res1 = p.preprocess("The prime minister didn't step down.")
    res2 = p.preprocess("The prime minister did not step down.")
    assert "not" in res1
    assert "not" in res2
    # Both should reduce to the same semantic representation
    assert "minister" in res1 and "step" in res1 and "not" in res1


def test_various_negation_terms():
    p = TextPreprocessor(preserve_negation=True)
    text = "We have never seen nobody nowhere without evidence."
    cleaned = p.preprocess(text)
    assert "never" in cleaned
    assert "without" in cleaned


# ── 4. Numbers and Percentages Tests ──────────────────────────────────────────

def test_number_preservation():
    p_keep = TextPreprocessor(preserve_numbers=True)
    p_drop = TextPreprocessor(preserve_numbers=False)

    text = "Inflation dropped by 5% in 2024 to 3.2 percent."
    cleaned_keep = p_keep.preprocess(text)
    cleaned_drop = p_drop.preprocess(text)

    assert "5" in cleaned_keep
    assert "2024" in cleaned_keep
    assert "5" not in cleaned_drop
    assert "2024" not in cleaned_drop


# ── 5. Publisher Dateline & Artifact Stripping Tests ─────────────────────────

def test_reuters_dateline_stripping():
    sample = "WASHINGTON (Reuters) - The U.S. Senate voted on the infrastructure bill."
    stripped = strip_publisher_artifacts(sample)
    assert "(Reuters)" not in stripped
    assert "WASHINGTON (Reuters) -" not in stripped
    assert "senate" in stripped.lower()
    assert "infrastructure" in stripped.lower()


def test_fake_news_artifact_stripping():
    fake_sample = "Secret bunker uncovered! Featured image via conservativepatriot. (Tweet)"
    stripped = strip_publisher_artifacts(fake_sample)
    assert "Featured image via" not in stripped
    assert "(Tweet)" not in stripped


def test_publisher_cue_detector():
    cues = check_publisher_cues_in_text("WASHINGTON (Reuters) - Spokesman confirmed.")
    assert "reuters" in cues
    assert "washington" in cues


# ── 6. Unseen Vocabulary Tests ───────────────────────────────────────────────

def test_unseen_vocabulary_transform():
    extractor = FeatureExtractor(max_features=50)
    train_docs = ["economic growth and trade agreements", "scientific discoveries in biology and space"]
    extractor.fit(train_docs)

    unseen_doc = ["quantum xylophone zzzzz unknownword12345"]
    feats = extractor.transform(unseen_doc)
    assert feats.shape == (1, len(extractor.get_feature_names()))
    assert feats.nnz == 0  # All zeros for completely unseen vocabulary


# ── 7. Missing Model Files & Incorrect Columns Tests ─────────────────────────

def test_missing_model_file_handling():
    with pytest.raises(Exception):
        predict_news("A standard news story about politics.", model_type='nonexistent_model_xyz')


def test_incorrect_dataset_columns():
    bad_df = pd.DataFrame({'wrong_col_1': [1, 2], 'wrong_col_2': ['a', 'b']})
    preprocessor = TextPreprocessor()
    with pytest.raises(KeyError):
        preprocessor.preprocess_dataframe(bad_df, text_column='nonexistent_text_col')


# ── 8. Model Serialization and Deserialization Tests ──────────────────────────

def test_classical_model_serialization_consistency():
    X_train = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.8],
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.9],
    ])
    y_train = np.array([0, 1, 0, 1])

    model1 = FakeNewsModel(model_type='logistic_regression')
    model1.train(X_train, y_train)
    preds1 = model1.predict(X_train)
    probs1 = model1.predict_proba(X_train)

    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model1.save(tmp_path)
        model2 = FakeNewsModel(model_type='logistic_regression')
        model2.load(tmp_path)
        preds2 = model2.predict(X_train)
        probs2 = model2.predict_proba(X_train)

        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_allclose(probs1, probs2, rtol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_deep_learning_serialization_consistency():
    texts = [
        "President signs law in capital.",
        "Shocking secret alien conspiracy exposed!",
        "Central bank sets benchmark lending rate.",
        "Bizarre scandal rocks elite establishment!",
    ] * 4
    labels = [0, 1, 0, 1] * 4

    lstm1 = LSTMModel(max_words=100, max_len=30, embedding_dim=16, hidden_dim=8)
    lstm1.fit(texts, labels, epochs=1, batch_size=4, verbose=False)

    p1 = lstm1.predict(texts[:4])
    pr1 = lstm1.predict_proba(texts[:4])

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_m, \
         tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_t:
        m_path = tmp_m.name
        t_path = tmp_t.name

    try:
        lstm1.save(m_path, t_path)
        lstm2 = LSTMModel(max_words=100, max_len=30, embedding_dim=16, hidden_dim=8)
        lstm2.load(m_path, t_path)

        p2 = lstm2.predict(texts[:4])
        pr2 = lstm2.predict_proba(texts[:4])

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_allclose(pr1, pr2, atol=1e-4)
    finally:
        for f in (m_path, t_path):
            if os.path.exists(f):
                os.remove(f)


# ── 9. Evaluation Metrics & Calibration Tests ────────────────────────────────

def test_evaluation_metrics_and_calibration():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.85, 0.65, 0.7, 0.3, 0.4])

    evaluator = ModelEvaluator(y_true, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    assert 'accuracy' in metrics
    assert 'macro_f1' in metrics
    assert 'pr_auc' in metrics
    assert 'roc_auc' in metrics
    assert 'brier_score' in metrics
    assert 'ece' in metrics

    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['macro_f1'] <= 1.0
    assert 0.0 <= metrics['brier_score'] <= 1.0
    assert metrics['true_positives'] == 3
    assert metrics['true_negatives'] == 3
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1


def test_cross_validation_metrics_calculation():
    X = np.random.randn(20, 5)
    y = np.array([0, 1] * 10)
    model = LogisticRegression(max_iter=100)

    cv_res = ModelEvaluator.cross_validate_metrics(model, X, y, cv=3, random_state=42)
    assert 'cv_accuracy_mean' in cv_res
    assert 'cv_accuracy_std' in cv_res
    assert 'cv_macro_f1_mean' in cv_res
    assert isinstance(cv_res['cv_accuracy_mean'], float)


def test_heuristic_risk_flags_empirical_evaluation():
    df = pd.DataFrame({
        'text': [
            "Official statement from government ministry regarding budget.",
            "BREAKING!!! SHOCKING SECRET CONSPIRACY EXPOSED!!! BUY NOW!!!",
            "Scientific report details observations from marine biology institute.",
            "OUTRAGEOUS HOAX UNBELIEVABLE SCANDAL LIES DETECTED!!!",
        ],
        'label': [0, 1, 0, 1],
    })

    flag_eval = evaluate_heuristic_risk_flags(df)
    assert 'has_exclamation_flag' in flag_eval
    assert 'any_flag' in flag_eval
    assert flag_eval['any_flag']['fake_trigger_rate'] >= 0.5
