import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_top_contributing_features(model, features, feature_names, top_n=10):
    if not hasattr(model, 'coef_'):
        return None

    coef = model.coef_[0]
    dense_features = features.toarray()[0]
    contributions = coef * dense_features

    indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
    results = []
    for idx in indices:
        if dense_features[idx] == 0:
            continue
        results.append({
            'feature': feature_names[idx],
            'contribution': float(contributions[idx]),
            'direction': 'fake' if contributions[idx] > 0 else 'real',
        })
    return results[:top_n]


def explain_with_shap(model, features, feature_names, top_n=10):
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
        indices = np.argsort(np.abs(values))[-top_n:][::-1]

        results = []
        for idx in indices:
            if dense_features[idx] == 0:
                continue
            results.append({
                'feature': feature_names[idx],
                'contribution': float(values[idx]),
                'direction': 'fake' if values[idx] > 0 else 'real',
            })
        return results[:top_n]
    except Exception:
        return get_top_contributing_features(model, features, feature_names, top_n)
