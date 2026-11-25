# Fake News Detection & Generalization Forensics

An experimentally rigorous natural language processing (NLP) study investigating machine learning classifiers for fake news detection, with a focus on **dataset leakage mitigation**, **cross-domain generalization on unseen publishers**, **negation-aware preprocessing**, and **calibrated probability estimates**.

---

## Key Research Findings & Contributions

1. **Dataset Bias & Publisher Shortcut Leakage**:
   - In standard ISOT data, **100% of real articles contain Reuters datelines or agency markers**, while **0% of fake articles do**.
   - A single-feature rule checking purely for the word `"Reuters"` achieves near-perfect accuracy without reading any news claims.
   - Models trained with publisher datelines stripped achieve higher generalizability when tested against completely unseen news publishers.
2. **Generalization Gap on Independent Sources**:
   - In-domain test accuracy reaches ~99–100% on random splits of ISOT.
   - When evaluated on an independent external benchmark with articles from varied publishers (AP, BBC, NPR, PolitiFact, and satirical/conspiracy sites), accuracy drops to ~75–80%, exposing the **generalization drop** inherent in single-dataset benchmarks.
3. **Negation & Number Preservation**:
   - Standard stopword removal strips `"not"`, converting *"The minister did not resign"* into *"minister resign"*, reversing sentence meaning.
   - The refactored preprocessor expands contractions, explicitly preserves negation operators (`not`, `no`, `never`, `cannot`, `without`), and retains numbers/percentages/dates essential for factual grounding.
4. **Calibrated Confidence & Responsible Interface**:
   - Uncalibrated neural or margin classifiers output extreme probabilities (e.g. 99%) that do not reflect true posterior correctness.
   - The interface displays **"Model Confidence Estimate"** with probability calibration and enforces an **Insufficient Information** state for short inputs (<15 words).
   - Prominent disclaimers clarify that the system detects stylistic and lexical correlation with training data, **not real-world factuality or truth**.

---

## Benchmark Results

### 1. Generalization Gap: In-Domain Test vs. Independent External Benchmark

| Architecture | In-Domain Accuracy | In-Domain Macro F1 | External Benchmark Accuracy | External Macro F1 | Accuracy Drop ($\Delta$) |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 100.0% | 1.0000 | 75.0% | 0.7494 | -25.0% |
| **Linear SVM** | 100.0% | 1.0000 | 75.0% | 0.7494 | -25.0% |
| **Random Forest** | 100.0% | 1.0000 | 70.0% | 0.6970 | -30.0% |
| **Gradient Boosting** | 100.0% | 1.0000 | 70.0% | 0.6970 | -30.0% |
| **Soft-Voting Ensemble** | 100.0% | 1.0000 | 75.0% | 0.7494 | -25.0% |
| **Bidirectional LSTM (Word Embeddings)** | 98.2% | 0.9820 | 70.0% | 0.6840 | -28.2% |
| **Self-Attention Transformer** | 98.5% | 0.9850 | 75.0% | 0.7420 | -23.5% |

### 2. Publisher Leakage Ablation Study

| Experiment Condition | In-Domain Test (With Cues) | In-Domain Test (Cues Stripped) | External Benchmark (Unseen) |
|---|---|---|---|
| **Single-Word Heuristic (`"reuters"` present)** | 100.0% | 0.0% | 50.0% (random guess) |
| **Model A (Trained on raw text with datelines)** | 100.0% | 98.0% | 75.0% |
| **Model B (Trained on text with datelines stripped)** | 100.0% | 100.0% | **80.0%** |

*Models trained on stripped text generalize substantially better to real-world unseen news outlets.*

---

## Project Structure

```
fake-news-detection/
├── app/
│   └── app.py                     # Streamlit interface with calibration & disclaimers
├── data/
│   ├── sample_isot.csv            # Representative ISOT sample for instant reproducibility
│   └── external_benchmark.csv     # Independent multi-source test set (BBC, AP, NPR, Hoaxes)
├── models/                        # Saved artifacts, joblib models, JSON reports, PNG plots
│   ├── results.json               # Full test metrics across all architectures
│   ├── experiment_config.json     # Experiment configuration, seeds & parameters
│   ├── leakage_report.json        # Publisher cue leakage & ablation results
│   ├── external_eval_results.json # Unseen dataset generalization results
│   ├── comparison.png             # Performance metric bar chart
│   ├── leakage_comparison.png     # Leakage ablation plot
│   └── external_generalization.png# Generalization drop comparison plot
├── src/
│   ├── preprocessing.py           # Negation & number preservation, dateline stripping
│   ├── feature_extraction.py      # Sublinear TF-IDF unigrams & bigrams
│   ├── models.py                  # Classical ML (LR baseline, Linear SVM, RF, GB, Ensemble)
│   ├── deep_learning.py           # PyTorch BiLSTM and Self-Attention Transformer
│   ├── explainability.py          # SHAP attribution, directionality & shortcut detection
│   ├── linguistic_features.py     # Heuristic risk flags & empirical validation
│   ├── evaluation.py              # Macro F1, PR-AUC, ROC-AUC, Brier score, ECE & CV
│   ├── leakage_experiments.py     # Publisher artifact frequency & single-cue test
│   ├── external_eval.py           # External benchmark evaluation pipeline
│   ├── train.py                   # Main training pipeline with cross-validation
│   └── predict.py                 # CLI inference with sufficiency & leakage guards
├── tests/
│   └── test_fake_news_detection.py# Pytest test suite (20 automated tests)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

Requires Python 3.10+ (tested up to Python 3.13):

```bash
git clone https://github.com/shashivardhan-m/fake-news-detection.git
cd fake-news-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Run Publisher Leakage Experiments
Demonstrates dataset shortcut bias and evaluates the single-cue heuristic:
```bash
python src/leakage_experiments.py
```

### 2. Train Models with Rigorous Evaluation
Trains classical models, ensembles, and deep learning architectures with 5-fold cross-validation, PR-AUC, Macro F1, and automatic external benchmark evaluation:
```bash
# Standard training (classical models + ensemble + cross-validation)
python src/train.py

# Include PyTorch BiLSTM and Transformer architectures
python src/train.py --train-lstm --train-transformer --epochs 5

# Replicate dataset bias for ablation comparison (without dateline stripping)
python src/train.py --no-strip-cues
```

### 3. Evaluate on Independent External Datasets
Evaluates trained models on unseen news publishers and reports the generalization gap:
```bash
python src/external_eval.py --benchmark data/external_benchmark.csv
```

### 4. Command-Line Prediction with Guardrails
Predicts label with confidence estimate, input sufficiency check, and SHAP evidence:
```bash
python src/predict.py "WASHINGTON (Reuters) - The senate approved a budget resolution today." --model logistic_regression --explain
```

### 5. Launch the Streamlit Forensic Web App
```bash
streamlit run app/app.py
```

### 6. Run the Automated Test Suite
Runs all 20 tests verifying negation preservation, number handling, dateline stripping, unseen vocabulary, empty inputs, serialization consistency, and metric correctness:
```bash
pytest tests/ -v
```

---

## Limitations & Ethical Considerations

- **Statistical Association vs. Factuality**: A machine learning model identifies stylistic alignment and vocabulary correlations present in training sets. It cannot query primary documents, verify event occurrences, or confirm objective fact.
- **Short Input Degradation**: Predictions on single headlines or isolated sentences lack surrounding paragraph context and exhibit higher variance.
- **Heuristic Risk Indicators**: Surface-level stylistic indicators (e.g. exclamation marks, capitalization, short sentences) correlate loosely with sensationalism but trigger false positives on legitimate breaking news. They must never be treated as definitive proof of disinformation.
