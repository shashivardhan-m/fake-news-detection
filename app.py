import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import FakeNewsModel

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.title("📰 Advanced Fake News Detection System")
st.markdown("---")

@st.cache_resource
def load_models():
    models = {}
    preprocessor = TextPreprocessor()
    
    extractor = FeatureExtractor()
    extractor.load('models/tfidf_vectorizer.joblib')
    
    lr_model = FakeNewsModel()
    lr_model.load('models/logistic_regression.joblib')
    models['Logistic Regression'] = {'model': lr_model, 'extractor': extractor}
    
    rf_model = FakeNewsModel(model_type='random_forest')
    rf_model.load('models/random_forest.joblib')
    models['Random Forest'] = {'model': rf_model, 'extractor': extractor}
    
    with open('models/results.json', 'r') as f:
        results = json.load(f)
    
    return preprocessor, models, results

try:
    preprocessor, models, results = load_models()
    
    tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Comparison", "ℹ️ About"])
    
    with tab1:
        st.header("Check if News is Real or Fake")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_choice = st.selectbox("Choose Model", list(models.keys()))
            text_input = st.text_area("News Article Text", height=300, placeholder="Paste your news article here...")
        
        with col2:
            st.subheader("Prediction Result")
            if st.button("Check News", type="primary"):
                if text_input.strip() == "":
                    st.warning("Please enter some text to check!")
                else:
                    with st.spinner("Analyzing..."):
                        cleaned_text = preprocessor.preprocess(text_input)
                        model_data = models[model_choice]
                        model = model_data['model']
                        extractor = model_data['extractor']
                        
                        features = extractor.transform([cleaned_text])
                        prediction = model.predict(features)[0]
                        probability = model.predict_proba(features)[0]
                        
                        label = "Fake" if prediction == 1 else "Real"
                        color = "red" if prediction == 1 else "green"
                        
                        st.markdown(f"## Prediction: <span style='color:{color}; font-size: 32px;'>{label}</span>", unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Real Probability", f"{probability[0]:.2%}")
                        with col_b:
                            st.metric("Fake Probability", f"{probability[1]:.2%}")
                        
                        st.progress(int(max(probability) * 100))
    
    with tab2:
        st.header("Model Performance Comparison")
        
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)
        
        st.subheader("Performance Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            sns.barplot(x=results_df.index, y=results_df[metric], ax=ax, palette='viridis')
            ax.set_title(f'{name} Comparison')
            ax.set_ylim([0.9, 1.0])
            ax.tick_params(axis='x', rotation=45)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.header("About This Project")
        st.markdown("""
        ## Advanced Fake News Detection System
        
        This project uses multiple machine learning models to detect fake news:
        
        ### Models Available:
        1. **Logistic Regression** with TF-IDF features
        2. **Random Forest** with TF-IDF features
        
        ### Features:
        - Multiple model comparison
        - Real-time predictions
        - Detailed performance metrics
        - Interactive visualizations
        - Easy-to-use interface
        
        ### Dataset:
        ISOT Fake News Dataset containing both real and fake news articles.
        """)

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Please train the models first by running 'python train.py'")

