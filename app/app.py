import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_dir, 'src'))

from explainability import explain_prediction_breakdown
from feature_extraction import FeatureExtractor
from linguistic_features import analyze_text, get_risk_flags
from models import FakeNewsModel, MODEL_CONFIGS
from predict import check_input_sufficiency, check_publisher_cues_in_text, FACT_CHECK_DISCLAIMER
from preprocessing import TextPreprocessor

st.set_page_config(
    page_title="Veritas — Fake News & Generalization Forensics",
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
        --warn:      #ffb020;
    }

    .stApp { background: var(--bg); color: var(--text); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    h1, h2, h3, .hero-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        letter-spacing: -0.02em;
    }
    p, label, .stMarkdown, .stCaption, span, div {
        font-family: 'DM Sans', sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #141922 0%, #1a2030 60%, #0f1520 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 0.4rem 0;
        line-height: 1.15;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        color: var(--muted);
        font-size: 0.95rem;
        margin: 0;
        max-width: 650px;
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

    .verdict-card { text-align: center; padding: 1.8rem 1.5rem; }
    .verdict-real {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--real);
        line-height: 1;
        margin: 0.25rem 0;
    }
    .verdict-fake {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--fake);
        line-height: 1;
        margin: 0.25rem 0;
    }
    .verdict-conf { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }

    .disclaimer-box {
        background: rgba(74,158,255,0.08);
        border: 1px solid rgba(74,158,255,0.25);
        border-left: 3px solid var(--blue);
        color: #b0d4ff;
        padding: 0.75rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .warning-box {
        background: rgba(255,176,32,0.08);
        border: 1px solid rgba(255,176,32,0.3);
        border-left: 3px solid var(--warn);
        color: #ffd880;
        padding: 0.75rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .flag-item {
        background: rgba(240,92,92,0.08);
        border: 1px solid rgba(240,92,92,0.25);
        border-left: 3px solid var(--fake);
        color: #f0a0a0;
        padding: 0.55rem 0.9rem;
        border-radius: 0 8px 8px 0;
        margin: 0.35rem 0;
        font-size: 0.82rem;
    }
    .flag-ok {
        background: rgba(62,207,142,0.08);
        border: 1px solid rgba(62,207,142,0.25);
        color: #7eeab8;
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        font-size: 0.82rem;
    }

    .model-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .tag-real {
        display: inline-block;
        background: rgba(62,207,142,0.15);
        color: var(--real);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
    }
    .tag-fake {
        display: inline-block;
        background: rgba(240,92,92,0.15);
        color: var(--fake);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_ARTICLES = {
    "Real — Reuters style (with dateline)": (
        "WASHINGTON (Reuters) - The Federal Reserve held benchmark interest rates steady on Wednesday, "
        "citing ongoing progress toward its 2 percent inflation target while noting that the labor market "
        "remains resilient. Chair Jerome Powell said policymakers would assess incoming economic data "
        "before deciding on future monetary adjustments."
    ),
    "Real — Independent (BBC, unseen source)": (
        "The Bank of England announced an increase in borrowing costs by a quarter percentage point on Thursday. "
        "Governor Andrew Bailey stated that bringing inflation back to the target remains the primary objective "
        "despite sluggish economic output across manufacturing sectors."
    ),
    "Fake — Sensational Clickbait": (
        "BREAKING: SHOCKING SECRET EXPOSED! Corrupt politicians caught in underground bunker hiding "
        "classified files! The mainstream media refuses to report this bombshell! Share this immediately "
        "before Twitter deletes it!"
    ),
    "Fake — Subtle Conspiracy": (
        "Confidential whistleblower memos confirm that telecommunication frequencies operate on resonant cognitive "
        "harmonics intended to pacify metropolitan populations. Aluminum shielding is the only effective defense "
        "against ambient radiation, according to alternative research groups."
    ),
    "Short input test (insufficient)": "The minister did not resign.",
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
}


def confidence_descriptor(prob):
    max_p = max(prob)
    if max_p >= 0.90:
        return "High Confidence Estimate", "●"
    if max_p >= 0.75:
        return "Moderate-High Confidence", "●"
    if max_p >= 0.60:
        return "Moderate Confidence (Uncertain)", "●"
    return "Low Confidence (Near Decision Boundary)", "●"


@st.cache_resource
def load_all_models():
    models_dir = os.path.join(project_dir, 'models')
    models = {}
    preprocessor = TextPreprocessor(preserve_negation=True, preserve_numbers=True, strip_publisher_cues=True)

    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    if not os.path.exists(vectorizer_path):
        return preprocessor, models, None, {}

    extractor = FeatureExtractor()
    extractor.load(vectorizer_path)

    # Classical models
    for model_type in list(MODEL_CONFIGS.keys()) + ['ensemble']:
        path = os.path.join(models_dir, FakeNewsModel.get_filename(model_type))
        if os.path.exists(path):
            m = FakeNewsModel(model_type=model_type)
            m.load(path)
            name = FakeNewsModel.get_display_name(model_type)
            models[name] = {'model': m, 'extractor': extractor, 'type': 'sklearn'}

    # Deep learning models
    dl_models = {}
    lstm_pt = os.path.join(models_dir, 'lstm_model.pt')
    if os.path.exists(lstm_pt):
        from deep_learning import LSTMModel
        lstm = LSTMModel()
        lstm.load(lstm_pt, os.path.join(models_dir, 'lstm_tokenizer.joblib'))
        dl_models['Bidirectional LSTM'] = lstm

    trans_pt = os.path.join(models_dir, 'transformer_model.pt')
    if os.path.exists(trans_pt):
        from deep_learning import TransformerModel
        trans = TransformerModel()
        trans.load(trans_pt, os.path.join(models_dir, 'transformer_tokenizer.joblib'))
        dl_models['Self-Attention Transformer'] = trans

    return preprocessor, models, extractor, dl_models


def load_json_file(filename):
    path = os.path.join(project_dir, 'models', filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


try:
    preprocessor, models, extractor, dl_models = load_all_models()
    results = load_json_file('results.json') or {}
    leakage_report = load_json_file('leakage_report.json')
    external_report = load_json_file('external_eval_results.json')
    experiment_config = load_json_file('experiment_config.json')

    if not models and not dl_models:
        st.markdown("""
        <div class="hero">
            <div class="hero-title">Veritas <span>Fake News Forensics</span></div>
            <p class="hero-sub">No trained models found. Run training first.</p>
        </div>
        """, unsafe_allow_html=True)
        st.code("python src/train.py", language="bash")
        st.stop()

    model_options = list(models.keys()) + list(dl_models.keys())

    # Header
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">NLP · Generalization Forensics · Bias Mitigation</div>
        <div class="hero-title">Veritas <span>Fake News Classifier</span></div>
        <p class="hero-sub">
            Rigorous misinformation classification emphasizing cross-domain generalization,
            publisher cue debiasing, and calibrated probability estimates.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙ Settings & Controls")
        model_choice = st.selectbox(
            "Active classifier",
            model_options,
            index=model_options.index('Ensemble (Voting)') if 'Ensemble (Voting)' in model_options else 0,
        )
        strip_cues_toggle = st.toggle("Strip publisher datelines / shortcuts", value=True, help="Removes Reuters and agency signatures to evaluate claim patterns without shortcut leakage.")
        show_shap = st.toggle("SHAP feature attribution", value=True)

        st.divider()
        st.markdown("**Sample Articles**")
        sample_choice = st.radio("Load test case:", ["None"] + list(SAMPLE_ARTICLES.keys()), label_visibility="collapsed")

    tab1, tab2, tab3, tab4 = st.tabs(["Analyze Article", "Model Comparison", "Generalization & Bias Analytics", "Project Methodology"])

    # ── Tab 1: Analyze ────────────────────────────────────────────────────────
    with tab1:
        col_in, col_out = st.columns([1.3, 1], gap="large")

        with col_in:
            st.markdown('<div class="card-label">Article Text Input</div>', unsafe_allow_html=True)
            default_val = SAMPLE_ARTICLES.get(sample_choice, "") if sample_choice != "None" else ""
            input_text = st.text_area(
                "Article body",
                value=default_val,
                height=260,
                placeholder="Paste headline and article text here...",
                label_visibility="collapsed",
            )
            analyze_clicked = st.button("Analyze Article", type="primary", use_container_width=True)

            st.markdown(f'<div class="disclaimer-box">ℹ <strong>Notice:</strong> {FACT_CHECK_DISCLAIMER}</div>', unsafe_allow_html=True)

        with col_out:
            st.markdown('<div class="card-label">Prediction & Calibration</div>', unsafe_allow_html=True)
            if analyze_clicked:
                if not input_text.strip():
                    st.warning("Please enter article text to analyze.")
                else:
                    is_suff, word_cnt, length_warn = check_input_sufficiency(input_text)
                    if not is_suff:
                        st.markdown(f'<div class="warning-box">⚠ <strong>Insufficient Information State:</strong> {length_warn}</div>', unsafe_allow_html=True)

                    cues = check_publisher_cues_in_text(input_text)
                    if cues and not strip_cues_toggle:
                        st.markdown(f'<div class="warning-box">⚠ <strong>Publisher Cue Warning:</strong> Found source markers: <code>{cues}</code>. Model may rely on publisher shortcuts.</div>', unsafe_allow_html=True)

                    prep = TextPreprocessor(preserve_negation=True, preserve_numbers=True, strip_publisher_cues=strip_cues_toggle)
                    cleaned = prep.preprocess(input_text)

                    if model_choice in dl_models:
                        pred = dl_models[model_choice].predict([cleaned])[0]
                        proba = dl_models[model_choice].predict_proba([cleaned])[0]
                    else:
                        m_data = models[model_choice]
                        feat = m_data['extractor'].transform([cleaned])
                        pred = m_data['model'].predict(feat)[0]
                        proba = m_data['model'].predict_proba(feat)[0]

                    label = "Fake" if pred == 1 else "Real"
                    conf_desc, _ = confidence_descriptor(proba)
                    css = "verdict-real" if label == "Real" else "verdict-fake"

                    st.markdown(f"""
                    <div class="card verdict-card">
                        <div class="card-label">Classification Verdict</div>
                        <div class="{css}">{label}</div>
                        <div class="verdict-conf">Model Confidence Estimate: <strong>{max(proba):.1%}</strong></div>
                        <div style="font-size:0.75rem;color:var(--muted);margin-top:0.3rem">{conf_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">Probability Distribution</div>
                        <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:0.2rem">
                            <span>Real: <strong style="color:var(--real)">{proba[0]:.1%}</strong></span>
                            <span>Fake: <strong style="color:var(--fake)">{proba[1]:.1%}</strong></span>
                        </div>
                        <div style="height:8px;background:var(--surface2);border-radius:999px;overflow:hidden">
                            <div style="width:{proba[0]*100}%;height:100%;background:var(--real);float:left"></div>
                            <div style="width:{proba[1]*100}%;height:100%;background:var(--fake);float:left"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if show_shap and model_choice in models:
                        inner = models[model_choice]['model'].model
                        if hasattr(inner, 'calibrated_classifiers_'):
                            inner = inner.calibrated_classifiers_[0].estimator
                        breakdown = explain_prediction_breakdown(inner, feat, extractor.get_feature_names(), top_n=6)

                        st.markdown('<div class="card-label" style="margin-top:1rem">Feature Evidence Alignment (SHAP)</div>', unsafe_allow_html=True)
                        if breakdown['has_publisher_shortcut_risk']:
                            st.caption(f"⚠ Top features contain publisher dateline cues: {breakdown['publisher_cues_found']}")

                        c_real, c_fake = st.columns(2)
                        with c_real:
                            st.caption("**Words Supporting Real**")
                            for w in breakdown['real_evidence'][:4]:
                                st.markdown(f"- <code>{w['feature']}</code> ({w['contribution']:.3f})", unsafe_allow_html=True)
                        with c_fake:
                            st.caption("**Words Supporting Fake**")
                            for w in breakdown['fake_evidence'][:4]:
                                st.markdown(f"- <code>{w['feature']}</code> ({w['contribution']:+.3f})", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:3rem 1rem;color:var(--muted)">
                    <div style="font-size:2rem;margin-bottom:0.5rem">📄</div>
                    <div>Enter text and click <strong style="color:var(--accent)">Analyze Article</strong></div>
                </div>
                """, unsafe_allow_html=True)

        if input_text.strip():
            st.markdown("---")
            stats = analyze_text(input_text)
            flags = get_risk_flags(stats)

            st.markdown('<div class="card-label">Stylistic & Forensic Indicators (Heuristics Only)</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Word Count", stats['word_count'])
            s2.metric("Sensational Lexicon", f"{stats['sensational_score']:.3f}")
            s3.metric("Caps Proportion", f"{stats['caps_ratio']:.1%}")
            s4.metric("Avg Sentence Length", stats['avg_sentence_length'])

            if flags:
                for fl in flags:
                    st.markdown(f'<div class="flag-item">⚠ {fl}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag-ok">✓ No overt sensationalist or clickbait punctuation detected.</div>', unsafe_allow_html=True)

    # ── Tab 2: Compare ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="card-label">Concurrent Multi-Model Prediction</div>', unsafe_allow_html=True)
        cmp_text = st.text_area("Article to compare across all architectures", height=130, placeholder="Paste text here...", label_visibility="collapsed")

        if st.button("Run Multi-Model Comparison", type="primary"):
            if not cmp_text.strip():
                st.warning("Enter text first.")
            else:
                prep = TextPreprocessor(preserve_negation=True, preserve_numbers=True, strip_publisher_cues=strip_cues_toggle)
                cleaned = prep.preprocess(cmp_text)
                cmp_rows = []

                for name, data in models.items():
                    feat = data['extractor'].transform([cleaned])
                    pred = data['model'].predict(feat)[0]
                    proba = data['model'].predict_proba(feat)[0]
                    cmp_rows.append({'name': name, 'pred': int(pred), 'proba': proba})

                for dl_name, dl_obj in dl_models.items():
                    pred = dl_obj.predict([cleaned])[0]
                    proba = dl_obj.predict_proba([cleaned])[0]
                    cmp_rows.append({'name': dl_name, 'pred': int(pred), 'proba': proba})

                cols = st.columns(min(len(cmp_rows), 3))
                for i, row in enumerate(cmp_rows):
                    lbl = 'Fake' if row['pred'] == 1 else 'Real'
                    tag = 'tag-fake' if lbl == 'Fake' else 'tag-real'
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div style="font-weight:600;font-size:0.9rem;margin-bottom:0.3rem">{row['name']}</div>
                            <span class="{tag}">{lbl}</span>
                            <div style="margin-top:0.4rem;font-size:0.78rem;color:var(--muted)">
                                Real: {row['proba'][0]:.1%} · Fake: {row['proba'][1]:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Tab 3: Analytics ────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="card-label">Generalization Gap & Dataset Bias Forensics</div>', unsafe_allow_html=True)

        if external_report:
            st.markdown("### 🌐 Generalization on Unseen External Benchmark")
            st.caption("Performance comparison on completely independent sources (BBC, AP, NPR, independent hoaxes):")

            ext_rows = []
            for m_name, d in external_report.items():
                m = d.get('metrics', {})
                ext_rows.append({
                    'Model': m_name,
                    'External Accuracy': m.get('accuracy', np.nan),
                    'External Macro F1': m.get('macro_f1', np.nan),
                    'PR-AUC': m.get('pr_auc', np.nan),
                    'Brier Score': m.get('brier_score', np.nan),
                })
            st.dataframe(pd.DataFrame(ext_rows).style.format({'External Accuracy': '{:.2%}', 'External Macro F1': '{:.4f}', 'PR-AUC': '{:.4f}', 'Brier Score': '{:.4f}'}), use_container_width=True)

        if leakage_report:
            st.markdown("### 🔍 Publisher Artifact Leakage Study")
            leak_stats = leakage_report.get('leakage_stats', {})
            h_base = leakage_report.get('heuristic_baseline', {})

            c1, c2, c3 = st.columns(3)
            c1.metric("Real with Reuters Marker", f"{leak_stats.get('real_with_reuters_pct', 0)}%")
            c2.metric("Fake with Reuters Marker", f"{leak_stats.get('fake_with_reuters_pct', 0)}%")
            c3.metric("Single-Cue Baseline Acc", f"{h_base.get('accuracy', 0):.1%}")

            st.caption("A naive 1-feature heuristic checking solely for 'Reuters' scores 100% on ISOT, proving severe publisher shortcut leakage in uncleaned datasets.")

        if results:
            st.markdown("### 📊 In-Domain Benchmark Metrics")
            res_df = pd.DataFrame(results).T
            display_cols = [c for c in ['accuracy', 'macro_f1', 'f1', 'pr_auc', 'roc_auc', 'cv_accuracy_mean', 'cv_macro_f1_mean'] if c in res_df.columns]
            st.dataframe(res_df[display_cols].style.format("{:.4f}"), use_container_width=True)

    # ── Tab 4: About ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        ### Research & Engineering Improvements

        This version addresses the core research challenges of fake news detection:
        1. **Publisher Cue Leakage**: Neutralizes agency datelines (e.g., `WASHINGTON (Reuters) -`) that let models achieve 99% accuracy via source shortcuts rather than factuality.
        2. **Negation & Number Preservation**: Expands contractions and explicitly retains negation tokens (`not`, `no`, `never`) and numbers/dates to avoid altering article semantics.
        3. **Generalization Gap Reporting**: Evaluates models on unseen external benchmarks (AP, BBC, NPR) and reports macro F1 and accuracy drops.
        4. **Multi-Architecture Comparison**: Benchmarks TF-IDF baselines against Linear SVM, Random Forests, Voting Ensembles, BiLSTMs, and Self-Attention Transformers.
        5. **Confidence Calibration**: Displays calibrated probability estimates with clear disclaimers that statistical classifiers do not verify factual truth.
        """)

except Exception as e:
    st.error(f"Application error: {e}")
    st.info("Train models first: `python src/train.py`")
