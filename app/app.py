import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_dir, 'src'))

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from linguistic_features import analyze_text, get_risk_flags
from explainability import explain_with_shap

st.set_page_config(
    page_title="Veritas — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:ital,wght@0,700;0,800;1,700&display=swap" rel="stylesheet">

<style>
    :root {
        --bg:        #0c0f14;
        --surface:   #141922;
        --surface2:  #1a2030;
        --border:    rgba(255,255,255,0.07);
        --text:      #e8eaf0;
        --muted:     #8892a4;
        --accent:    #e8a838;
        --real:      #3ecf8e;
        --fake:      #f05c5c;
        --blue:      #4a9eff;
    }

    .stApp { background: var(--bg); color: var(--text); }

    /* Hide default chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* Typography */
    h1, h2, h3, .hero-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        letter-spacing: -0.02em;
    }
    p, label, .stMarkdown, .stCaption, span, div {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #141922 0%, #1a2030 60%, #0f1520 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(232,168,56,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 0.4rem 0;
        line-height: 1.15;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
        max-width: 560px;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(232,168,56,0.12);
        border: 1px solid rgba(232,168,56,0.3);
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.75rem;
    }

    /* Stat pills in hero */
    .stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.25rem; }
    .stat-pill {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        min-width: 110px;
    }
    .stat-pill .val {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
    }
    .stat-pill .lbl {
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.75rem;
    }

    /* Verdict */
    .verdict-card {
        text-align: center;
        padding: 2rem 1.5rem;
    }
    .verdict-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }
    .verdict-real {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 800;
        color: var(--real);
        line-height: 1;
        margin: 0.25rem 0;
    }
    .verdict-fake {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 800;
        color: var(--fake);
        line-height: 1;
        margin: 0.25rem 0;
    }
    .verdict-conf {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 0.5rem;
    }

    /* Probability bars */
    .prob-row { margin: 0.6rem 0; }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 0.3rem;
    }
    .prob-header .name { color: var(--muted); font-weight: 500; }
    .prob-header .pct  { font-weight: 700; }
    .prob-track {
        background: var(--surface2);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .prob-fill-real {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2a9d6e, var(--real));
        transition: width 0.6s ease;
    }
    .prob-fill-fake {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #c0392b, var(--fake));
        transition: width 0.6s ease;
    }

    /* Risk flags */
    .flag-item {
        background: rgba(240,92,92,0.08);
        border: 1px solid rgba(240,92,92,0.25);
        border-left: 3px solid var(--fake);
        color: #f0a0a0;
        padding: 0.55rem 0.9rem;
        border-radius: 0 8px 8px 0;
        margin: 0.35rem 0;
        font-size: 0.85rem;
    }
    .flag-ok {
        background: rgba(62,207,142,0.08);
        border: 1px solid rgba(62,207,142,0.25);
        color: #7eeab8;
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        font-size: 0.85rem;
    }

    /* Model compare cards */
    .model-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .model-card .model-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text);
        margin-bottom: 0.4rem;
    }
    .tag-real {
        display: inline-block;
        background: rgba(62,207,142,0.15);
        color: var(--real);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
    }
    .tag-fake {
        display: inline-block;
        background: rgba(240,92,92,0.15);
        color: var(--fake);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
    }

    /* Consensus banner */
    .consensus {
        background: linear-gradient(90deg, rgba(74,158,255,0.1), rgba(232,168,56,0.1));
        border: 1px solid rgba(74,158,255,0.25);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .consensus strong { color: var(--accent); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0d12 !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem;
        color: var(--accent);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: var(--surface);
        border-radius: 12px;
        padding: 0.35rem;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--muted);
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface2) !important;
        color: var(--accent) !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #0c0f14 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* Inputs */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Metrics override */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: var(--text) !important;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }

    /* Sample chips */
    .sample-chip {
        display: inline-block;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.78rem;
        color: var(--muted);
        cursor: pointer;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }

    /* About section */
    .about-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .about-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
    }
    .about-item h4 {
        font-family: 'Playfair Display', serif;
        color: var(--accent);
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
    }
    .about-item p {
        color: var(--muted);
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_ARTICLES = {
    "Real — Reuters style": (
        "WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday, "
        "citing ongoing progress on inflation while noting that the labor market remains resilient. "
        "Chair Jerome Powell said policymakers would assess incoming data before making further moves."
    ),
    "Fake — Clickbait": (
        "BREAKING!!! SHOCKING TRUTH EXPOSED!!! Scientists DON'T want you to know this ONE SECRET "
        "that the government is HIDING from you!!! You won't BELIEVE what happens next!!! "
        "Share this NOW before they DELETE it!!!"
    ),
    "Fake — Conspiracy": (
        "Exclusive report reveals a massive conspiracy involving world leaders and secret labs. "
        "Unbelievable documents prove everything you've been told is a complete lie. "
        "The mainstream media refuses to cover this urgent alert."
    ),
}

CHART_STYLE = {
    'figure.facecolor': '#141922',
    'axes.facecolor': '#141922',
    'axes.edgecolor': '#8892a4',
    'axes.labelcolor': '#e8eaf0',
    'text.color': '#e8eaf0',
    'xtick.color': '#8892a4',
    'ytick.color': '#8892a4',
    'grid.color': '#1a2030',
    'legend.facecolor': '#141922',
    'legend.edgecolor': '#1a2030',
}


def confidence_label(prob):
    max_prob = max(prob)
    if max_prob >= 0.9:
        return "Very High", "●"
    if max_prob >= 0.75:
        return "High", "●"
    if max_prob >= 0.6:
        return "Moderate", "●"
    return "Low", "●"


@st.cache_resource
def load_sklearn_models():
    models_dir = os.path.join(project_dir, 'models')
    preprocessor = TextPreprocessor()
    models = {}

    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    if not os.path.exists(vectorizer_path):
        return preprocessor, models, None

    extractor = FeatureExtractor()
    extractor.load(vectorizer_path)

    for model_type in list(MODEL_CONFIGS.keys()) + ['ensemble']:
        path = os.path.join(models_dir, FakeNewsModel.get_filename(model_type))
        if os.path.exists(path):
            model = FakeNewsModel(model_type=model_type)
            model.load(path)
            display_name = FakeNewsModel.get_display_name(model_type)
            models[display_name] = {'model': model, 'extractor': extractor, 'type': model_type}

    return preprocessor, models, extractor


@st.cache_resource
def load_lstm_model():
    models_dir = os.path.join(project_dir, 'models')
    model_path = os.path.join(models_dir, 'lstm_model.keras')
    if not os.path.exists(model_path):
        return None
    from deep_learning import LSTMModel
    lstm = LSTMModel()
    lstm.load(model_path, os.path.join(models_dir, 'lstm_tokenizer.joblib'))
    return lstm


def load_results():
    results_path = os.path.join(project_dir, 'models', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def run_prediction(text, model_choice, models, lstm_model, preprocessor):
    cleaned = preprocessor.preprocess(text)
    if model_choice == 'Bidirectional LSTM' and lstm_model:
        pred = lstm_model.predict([cleaned])[0]
        proba = lstm_model.predict_proba([cleaned])[0]
    else:
        model_data = models[model_choice]
        features = model_data['extractor'].transform([cleaned])
        pred = model_data['model'].predict(features)[0]
        proba = model_data['model'].predict_proba(features)[0]
    return cleaned, int(pred), proba


def render_verdict(label, proba, conf_text):
    css = "verdict-real" if label == "Real" else "verdict-fake"
    st.markdown(f"""
    <div class="card verdict-card">
        <div class="verdict-label">Verdict</div>
        <div class="{css}">{label}</div>
        <div class="verdict-conf">{conf_text} confidence · {max(proba):.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    real_pct = proba[0] * 100
    fake_pct = proba[1] * 100
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Probability Breakdown</div>
        <div class="prob-row">
            <div class="prob-header">
                <span class="name">Real</span>
                <span class="pct" style="color:var(--real)">{real_pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill-real" style="width:{real_pct:.1f}%"></div>
            </div>
        </div>
        <div class="prob-row">
            <div class="prob-header">
                <span class="name">Fake</span>
                <span class="pct" style="color:var(--fake)">{fake_pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill-fake" style="width:{fake_pct:.1f}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_linguistic_analysis(text):
    stats = analyze_text(text)
    flags = get_risk_flags(stats)

    st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Linguistic Signals</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", stats['word_count'])
    c2.metric("Sentences", stats['sentence_count'])
    c3.metric("Avg Length", stats['avg_sentence_length'])
    c4.metric("Sensational", f"{stats['sensational_score']:.3f}")

    if flags:
        st.markdown('<div class="card-label" style="margin-top:1rem">Risk Indicators</div>', unsafe_allow_html=True)
        for flag in flags:
            st.markdown(f'<div class="flag-item">⚠ {flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="flag-ok" style="margin-top:0.75rem">✓ No strong linguistic risk indicators</div>', unsafe_allow_html=True)


def render_explanations(model, features, extractor):
    inner_model = model.model
    if hasattr(inner_model, 'calibrated_classifiers_'):
        inner_model = inner_model.calibrated_classifiers_[0].estimator

    explanations = explain_with_shap(inner_model, features, extractor.get_feature_names())
    if not explanations:
        st.info("Explainability not available for this model.")
        return

    st.markdown('<div class="card-label">Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
    exp_df = pd.DataFrame(explanations)
    exp_df['impact'] = exp_df['contribution'].abs()
    exp_df = exp_df.sort_values('impact', ascending=True)

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors = ['#f05c5c' if d == 'fake' else '#3ecf8e' for d in exp_df['direction']]
        ax.barh(exp_df['feature'], exp_df['contribution'], color=colors, height=0.6)
        ax.set_xlabel('Contribution')
        ax.axvline(0, color='#8892a4', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_hero(model_count, has_lstm):
    total = model_count + (1 if has_lstm else 0)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">NLP · Ensemble ML · Explainable AI</div>
        <div class="hero-title">Veritas <span>News Analyzer</span></div>
        <p class="hero-sub">
            Classify articles as real or fake using multi-model ensemble voting,
            linguistic forensics, and SHAP feature attribution.
        </p>
        <div class="stat-row">
            <div class="stat-pill"><div class="val">{total}</div><div class="lbl">Models</div></div>
            <div class="stat-pill"><div class="val">TF-IDF</div><div class="lbl">Features</div></div>
            <div class="stat-pill"><div class="val">SHAP</div><div class="lbl">Explainability</div></div>
            <div class="stat-pill"><div class="val">Batch</div><div class="lbl">CSV Upload</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


try:
    preprocessor, models, extractor = load_sklearn_models()
    lstm_model = load_lstm_model()
    results = load_results()

    if not models and lstm_model is None:
        st.markdown("""
        <div class="hero">
            <div class="hero-title">Veritas <span>News Analyzer</span></div>
            <p class="hero-sub">No trained models found. Run training first.</p>
        </div>
        """, unsafe_allow_html=True)
        st.code("python src/train.py", language="bash")
        st.stop()

    model_options = list(models.keys())
    if lstm_model:
        model_options.append('Bidirectional LSTM')

    render_hero(len(models), lstm_model is not None)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙ Settings")
        model_choice = st.selectbox(
            "Active model",
            model_options,
            index=model_options.index('Ensemble (Voting)') if 'Ensemble (Voting)' in model_options else 0,
        )
        show_explain = st.toggle("SHAP explanations", value=True)
        st.divider()
        st.markdown("**Sample articles**")
        sample_choice = st.radio(
            "Load example",
            ["None"] + list(SAMPLE_ARTICLES.keys()),
            label_visibility="collapsed",
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Compare Models", "Analytics", "About"])

    # ── Tab 1: Analyze ────────────────────────────────────────────────────────
    with tab1:
        col_input, col_result = st.columns([1.4, 1], gap="large")

        with col_input:
            st.markdown('<div class="card-label">Article Input</div>', unsafe_allow_html=True)
            default_text = SAMPLE_ARTICLES.get(sample_choice, "") if sample_choice != "None" else ""
            text_input = st.text_area(
                "Paste article",
                value=default_text,
                height=300,
                placeholder="Paste a headline and article body here…",
                label_visibility="collapsed",
            )
            analyze_btn = st.button("Analyze Article", type="primary", use_container_width=True)

        with col_result:
            st.markdown('<div class="card-label">Analysis Result</div>', unsafe_allow_html=True)
            if analyze_btn:
                if not text_input.strip():
                    st.warning("Enter article text to analyze.")
                else:
                    with st.spinner("Running pipeline…"):
                        cleaned, pred, proba = run_prediction(
                            text_input, model_choice, models, lstm_model, preprocessor,
                        )
                        label = "Fake" if pred == 1 else "Real"
                        conf_text, _ = confidence_label(proba)
                        render_verdict(label, proba, conf_text)

                        if show_explain and model_choice != 'Bidirectional LSTM':
                            render_explanations(
                                models[model_choice]['model'],
                                models[model_choice]['extractor'].transform([cleaned]),
                                models[model_choice]['extractor'],
                            )
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:2.5rem 1rem;color:var(--muted)">
                    <div style="font-size:2rem;margin-bottom:0.5rem">📄</div>
                    <div style="font-size:0.9rem">Paste an article and click<br><strong style="color:var(--accent)">Analyze Article</strong></div>
                </div>
                """, unsafe_allow_html=True)

        if text_input.strip():
            st.markdown("---")
            render_linguistic_analysis(text_input)

        st.markdown("---")
        st.markdown('<div class="card-label">Batch Processing</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV with a `text` column", type=['csv'], label_visibility="collapsed")
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            if 'text' not in batch_df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                st.caption(f"{len(batch_df)} articles loaded")
                if st.button("Run batch analysis", use_container_width=True):
                    with st.spinner("Processing batch…"):
                        rows = []
                        for text in batch_df['text']:
                            _, pred, proba = run_prediction(
                                str(text), model_choice, models, lstm_model, preprocessor,
                            )
                            rows.append({
                                'prediction': 'Fake' if pred == 1 else 'Real',
                                'real_prob': round(proba[0], 4),
                                'fake_prob': round(proba[1], 4),
                            })
                        output_df = pd.concat([batch_df, pd.DataFrame(rows)], axis=1)
                        st.dataframe(output_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "↓ Download results CSV",
                            output_df.to_csv(index=False),
                            "veritas_predictions.csv",
                            "text/csv",
                            use_container_width=True,
                        )

    # ── Tab 2: Compare ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="card-label">Multi-Model Comparison</div>', unsafe_allow_html=True)
        compare_text = st.text_area(
            "Article to compare",
            height=140,
            placeholder="Enter text to run through all models simultaneously…",
            label_visibility="collapsed",
        )

        if st.button("Run comparison", type="primary"):
            if not compare_text.strip():
                st.warning("Enter text first.")
            else:
                cleaned = preprocessor.preprocess(compare_text)
                rows = []

                for name, data in models.items():
                    features = data['extractor'].transform([cleaned])
                    pred = data['model'].predict(features)[0]
                    proba = data['model'].predict_proba(features)[0]
                    rows.append({'name': name, 'pred': int(pred), 'proba': proba})

                if lstm_model:
                    pred = lstm_model.predict([cleaned])[0]
                    proba = lstm_model.predict_proba([cleaned])[0]
                    rows.append({'name': 'Bidirectional LSTM', 'pred': int(pred), 'proba': proba})

                cols = st.columns(min(len(rows), 3))
                for i, row in enumerate(rows):
                    label = 'Fake' if row['pred'] == 1 else 'Real'
                    tag = 'tag-fake' if label == 'Fake' else 'tag-real'
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-name">{row['name']}</div>
                            <span class="{tag}">{label}</span>
                            <div style="margin-top:0.5rem;font-size:0.78rem;color:var(--muted)">
                                Real {row['proba'][0]:.1%} · Fake {row['proba'][1]:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                fake_votes = sum(1 for r in rows if r['pred'] == 1)
                real_votes = len(rows) - fake_votes
                consensus = 'Fake' if fake_votes > real_votes else 'Real'
                st.markdown(f"""
                <div class="consensus">
                    <strong>Consensus: {consensus}</strong> — {fake_votes} model(s) flagged Fake,
                    {real_votes} flagged Real across {len(rows)} models.
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 3: Analytics ────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="card-label">Model Performance</div>', unsafe_allow_html=True)
        if results:
            results_df = pd.DataFrame(results).T
            st.dataframe(
                results_df.style.format("{:.4f}").background_gradient(cmap='YlOrRd', axis=0),
                use_container_width=True,
            )

            metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] if m in results_df.columns]
            if metrics:
                with plt.rc_context(CHART_STYLE):
                    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
                    if len(metrics) == 1:
                        axes = [axes]
                    palette = ['#e8a838', '#4a9eff', '#3ecf8e', '#f05c5c', '#a78bfa']
                    for ax, metric, color in zip(axes, metrics, palette):
                        bars = ax.bar(range(len(results_df)), results_df[metric], color=color, alpha=0.85)
                        ax.set_xticks(range(len(results_df)))
                        ax.set_xticklabels(results_df.index, rotation=35, ha='right', fontsize=7)
                        ax.set_title(metric.replace('_', ' ').title(), fontsize=10, pad=8)
                        ymin = max(0.8, results_df[metric].min() - 0.02)
                        ax.set_ylim([ymin, 1.0])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        for bar in bars:
                            ax.annotate(
                                f'{bar.get_height():.3f}',
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom', fontsize=7,
                            )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            comparison_img = os.path.join(project_dir, 'models', 'comparison.png')
            if os.path.exists(comparison_img):
                st.image(comparison_img, caption="Training comparison chart")
        else:
            st.warning("No results found. Run `python src/train.py` first.")

    # ── Tab 4: About ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="about-grid">
            <div class="about-item">
                <h4>Logistic Regression</h4>
                <p>TF-IDF bigrams with a balanced linear classifier. Fast and interpretable.</p>
            </div>
            <div class="about-item">
                <h4>Random Forest</h4>
                <p>200 bagged decision trees with class balancing for robust non-linear patterns.</p>
            </div>
            <div class="about-item">
                <h4>Gradient Boosting</h4>
                <p>Sequential tree boosting that captures subtle textual deception signals.</p>
            </div>
            <div class="about-item">
                <h4>Linear SVM</h4>
                <p>Calibrated max-margin classifier for high-confidence boundary decisions.</p>
            </div>
            <div class="about-item">
                <h4>Ensemble Voting</h4>
                <p>Soft-voting across LR, RF, and GB — recommended default for best reliability.</p>
            </div>
            <div class="about-item">
                <h4>Bidirectional LSTM</h4>
                <p>Optional deep learning model. Train with <code>--train-lstm</code> flag.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.code("""python src/train.py                  # Train all models
python src/train.py --train-lstm     # Include LSTM
python src/train.py --sample 5000    # Quick subset training
streamlit run app/app.py             # Launch this app""", language="bash")

except Exception as e:
    st.error(f"Application error: {e}")
    st.info("Train models first: `python src/train.py`")
