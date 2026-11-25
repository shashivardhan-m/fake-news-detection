import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Known publisher/source cue vocabulary that may indicate shortcut learning
KNOWN_PUBLISHER_CUES = {
    'reuters', 'washington', 'london', 'beijing', 'berlin', 'moscow',
    'tweet', 'twitter', 'facebook', 'image', 'photo', 'featured',
    'reporting', 'editing', 'spokesman', 'spokeswoman', 'source', 'sources',
    'wednesday', 'thursday', 'friday', 'monday', 'tuesday',
}


def get_top_contributing_features(model, features, feature_names, top_n=10):
    """
    Extracts linear model coefficients multiplied by feature values.
    Returns positive (Fake-leaning) and negative (Real-leaning) attributions.
    """
    if not hasattr(model, 'coef_'):
        return None

    coef = model.coef_[0]
    dense_features = features.toarray()[0]
    contributions = coef * dense_features

    non_zero_indices = np.where(dense_features != 0)[0]
    if len(non_zero_indices) == 0:
        return []

    sorted_indices = non_zero_indices[np.argsort(np.abs(contributions[non_zero_indices]))[::-1]]
    top_indices = sorted_indices[:top_n]

    results = []
    for idx in top_indices:
        val = float(contributions[idx])
        word = feature_names[idx]
        results.append({
            'feature': word,
            'contribution': val,
            'abs_contribution': abs(val),
            'direction': 'Fake' if val > 0 else 'Real',
            'is_publisher_cue': word.lower() in KNOWN_PUBLISHER_CUES,
        })
    return results


def explain_with_shap(model, features, feature_names, top_n=10):
    """
    SHAP-based feature importance with fallback to linear coefficients.
    """
    if not SHAP_AVAILABLE:
        return get_top_contributing_features(model, features, feature_names, top_n)

    try:
        if hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, features)
        elif hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            return get_top_contributing_features(model, features, feature_names, top_n)

        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        values = shap_values[0]
        dense_features = features.toarray()[0]
        non_zero = np.where(dense_features != 0)[0]

        if len(non_zero) == 0:
            return []

        sorted_indices = non_zero[np.argsort(np.abs(values[non_zero]))[::-1]]
        results = []
        for idx in sorted_indices[:top_n]:
            val = float(values[idx])
            word = feature_names[idx]
            results.append({
                'feature': word,
                'contribution': val,
                'abs_contribution': abs(val),
                'direction': 'Fake' if val > 0 else 'Real',
                'is_publisher_cue': word.lower() in KNOWN_PUBLISHER_CUES,
            })
        return results
    except Exception:
        return get_top_contributing_features(model, features, feature_names, top_n)


def explain_prediction_breakdown(model, features, feature_names, top_n=10):
    """
    Detailed explanation breakdown:
    - Evidence pushing towards Fake
    - Evidence pushing towards Real
    - Detection of publisher shortcut reliance
    - Crucial caveat: feature attribution reflects training corpus correlations, NOT factual truth.
    """
    explanations = explain_with_shap(model, features, feature_names, top_n=top_n * 2)
    if not explanations:
        return {
            'fake_evidence': [],
            'real_evidence': [],
            'publisher_cues_found': [],
            'disclaimer': (
                "Feature attribution shows which words in the text most closely matched "
                "patterns learned during training. It does NOT establish factual veracity."
            ),
        }

    fake_words = [item for item in explanations if item['direction'] == 'Fake'][:top_n]
    real_words = [item for item in explanations if item['direction'] == 'Real'][:top_n]
    publisher_cues = [item['feature'] for item in explanations if item['is_publisher_cue']]

    return {
        'fake_evidence': fake_words,
        'real_evidence': real_words,
        'publisher_cues_found': publisher_cues,
        'has_publisher_shortcut_risk': len(publisher_cues) > 0,
        'disclaimer': (
            "Feature attribution reflects statistical word correlations in the training dataset. "
            "Words pushing toward 'Real' or 'Fake' do not prove factuality or falsehood."
        ),
    }


def analyze_prediction_errors_and_explanations(
    model, features, feature_names, y_true, y_pred, texts=None, top_n=5, sample_n=3
):
    """
    Error analysis: inspects feature attributions for:
    - True Positives (Correct Fake)
    - True Negatives (Correct Real)
    - False Positives (Real misclassified as Fake)
    - False Negatives (Fake misclassified as Real)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    def inspect_samples(indices):
        cases = []
        for idx in indices[:sample_n]:
            feat_row = features[idx:idx+1]
            exp = explain_prediction_breakdown(model, feat_row, feature_names, top_n=top_n)
            cases.append({
                'index': int(idx),
                'snippet': texts[idx][:150] + '...' if texts is not None else None,
                'fake_cues': [f['feature'] for f in exp['fake_evidence'][:3]],
                'real_cues': [r['feature'] for r in exp['real_evidence'][:3]],
                'publisher_shortcuts': exp['publisher_cues_found'],
            })
        return cases

    return {
        'summary': {
            'true_positives': len(tp_idx),
            'true_negatives': len(tn_idx),
            'false_positives': len(fp_idx),
            'false_negatives': len(fn_idx),
        },
        'false_positives_analysis': inspect_samples(fp_idx),
        'false_negatives_analysis': inspect_samples(fn_idx),
        'correct_real_analysis': inspect_samples(tn_idx),
        'correct_fake_analysis': inspect_samples(tp_idx),
    }
