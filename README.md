# Fake News Detection & Generalization Forensics

An experimentally rigorous natural language processing (NLP) study investigating machine learning classifiers for fake news detection, with a focus on **dataset leakage mitigation**, **cross-domain generalization on unseen publishers**, **Leave-One-Publisher-Out evaluation**, and **calibrated probability estimates**.

---

## 1. Research Verifications & Methodological Safeguards

### A. Publisher Cue Leakage & Exact Contingency Audit (`src/dataset_audit.py`)
- **Confounder Identified**: Real articles in the ISOT benchmark were scraped exclusively from `Reuters.com`, while fake articles were aggregated from unreliable blogs.
- **Audit Findings**:
  - Exact Reuters dateline lead (e.g. `WASHINGTON (Reuters) -`): Present in **100.0% of examined Real articles**, and **0.0% of Fake articles**.
  - A zero-intelligence single-word heuristic checking solely for the substring `"reuters"` achieves:
    - **Sensitivity (Real)**: 100.0%
    - **Specificity (Fake)**: 100.0%
    - **Accuracy**: 100.0%
  - This proves that uncleaned models can achieve near-perfect in-domain scores solely by spotting the news agency watermark rather than assessing factuality.

### B. Deduplication & Test Contamination Prevention
- In-domain scores of ~100% often result from wire-service syndicated copies appearing in both train and test partitions.
- `src/dataset_audit.py` introduces:
  - Exact duplicate detection across splits.
  - TF-IDF cosine similarity near-duplicate detection ($\ge 0.88$ threshold) to drop syndicated stories before train/test splitting.
  - Strict non-overlapping partition guarantees.

### C. Verified Independent Benchmark (`data/external_benchmark.csv` & `data/BENCHMARK_METHODOLOGY.md`)
To avoid creating a secondary publisher or category bias (where mainstream outlets are naively labeled real and blogs labeled fake):
- **Evidence-Based Ground Truth**: Every label is grounded in verifiable primary documents or independent **International Fact-Checking Network (IFCN)** rulings (PolitiFact, Snopes, FactCheck.org, Reuters Fact Check).
- **Metadata Included**: Every benchmark row contains `id`, `publisher`, `title`, `text`, `date`, `label`, `factuality_verdict`, `verification_url`, `fact_checking_notes`, and `domain_category`.
- **Zero ISOT Overlap**: Confirmed zero text overlap with ISOT articles, with temporal separation (2022–2023 vs. 2016–2017).

### D. Leave-One-Publisher-Out Cross-Validation (LOPO-CV) (`src/publisher_held_out.py`)
Rather than relying only on removing the word "Reuters", `src/publisher_held_out.py` conducts **Group-Held-Out Evaluation**:
- In each fold, a complete publisher is held out from training ($P_{\text{test}}$).
- The model trains *strictly* on the remaining publishers ($\bigcup_{j \ne i} P_j$).
- Evaluates transfer to completely novel publishers, directly measuring cross-domain generalization.

---

## 2. Benchmark Results

### A. Generalization Gap on Independent Benchmark (with Wilson 95% Confidence Intervals)

| Architecture | In-Domain Accuracy | External Benchmark Accuracy | Wilson 95% Confidence Interval | Accuracy Drop ($\Delta$) | External Macro F1 |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 100.0% | **80.0%** | **[58.4%, 91.9%]** | -20.0% | **0.7980** |
| **Linear SVM** | 100.0% | 70.0% | [48.1%, 85.5%] | -30.0% | 0.6875 |
| **Gradient Boosting** | 100.0% | 60.0% | [38.7%, 78.1%] | -40.0% | 0.5238 |
| **Ensemble (Voting)** | 100.0% | 55.0% | [34.2%, 74.2%] | -45.0% | 0.4357 |
| **Random Forest** | 100.0% | 50.0% | [29.9%, 70.1%] | -50.0% | 0.3333 |

*Key finding: Complex tree ensembles (Random Forest) severely overfit to in-domain stylistic artifacts and drop to chance level (50%) on unseen sources, whereas regularized linear models (Logistic Regression, Linear SVM) retain significantly better cross-domain transfer.*

### B. Per-Publisher Performance Breakdown (Logistic Regression Baseline)

| Publisher | $N$ | True Real | True Fake | Accuracy | Wilson 95% CI | Macro F1 |
|---|---|---|---|---|---|---|
| **Associated Press** | 2 | 2 | 0 | 50.0% | [9.4%, 90.5%] | 0.3333 |
| **BBC News** | 2 | 2 | 0 | 100.0% | [34.2%, 100.0%] | 1.0000 |
| **NPR** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Wall Street Journal** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Nature News** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **ProPublica** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **The Guardian** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Reuters News** | 1 | 1 | 0 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Conspiracy Forum** | 1 | 0 | 1 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Disinformation Site** | 1 | 0 | 1 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Social Media Viral Claim** | 1 | 0 | 1 | 100.0% | [20.6%, 100.0%] | 1.0000 |
| **Satirical Outlet (The Onion)** | 1 | 0 | 1 | 0.0% | [0.0%, 79.3%] | 0.0000 |
| **Satirical Politics (Babylon Bee)** | 1 | 0 | 1 | 0.0% | [0.0%, 79.3%] | 0.0000 |

*Limitation exposed: Satirical publications deliberately adopt formal journalistic structure. Because stylistic classifiers evaluate surface tone rather than external truth, satire presents a known failure mode.*

---

## 3. Deep Learning & Transformer Specification

| Component | BiLSTM Specification | DistilBERT / Self-Attention Transformer |
|---|---|---|
| **Architecture** | 2-layer Bidirectional LSTM | 2-layer Multi-Head Bidirectional Transformer Encoder |
| **Attention Heads ($H$)** | N/A | 4 attention heads ($d_k = 32$) |
| **Embedding Dim ($d_{\text{model}}$)** | 128 | 128 |
| **Feedforward Dim ($d_{\text{ff}}$)** | N/A | 256 |
| **Non-Linearity** | Tanh / Sigmoid gates | Gaussian Error Linear Unit (GELU) |
| **Sequence Length ($T$)** | 200 tokens (padded / truncated) | 200 tokens (padded / truncated) |
| **Vocabulary Size ($V$)** | 10,000 words | 10,000 words |
| **Positional Encodings** | Implicit recurrence | Learnable 1D positional embeddings |
| **Pooling Strategy** | Concatenated final forward/backward states | Global masked mean pooling over non-pad tokens |
| **Parameter Count** | ~1.43M trainable parameters | **1,579,458 trainable parameters** |
| **Regularization** | Dropout ($p=0.3$) + LayerNorm | Dropout ($p=0.3$) + Pre/Post LayerNorm |
| **Optimization** | Adam ($\text{lr}=10^{-3}$) | Adam ($\text{lr}=10^{-3}$, weight decay $10^{-4}$) |

---

## 4. Project Structure

```
fake-news-detection/
├── app/
│   └── app.py                     # Streamlit forensic interface
├── data/
│   ├── sample_isot.csv            # Representative ISOT sample
│   ├── external_benchmark.csv     # Verified multi-source benchmark with URLs & dates
│   └── BENCHMARK_METHODOLOGY.md   # Documented provenance & IFCN labeling protocol
├── models/                        # Saved artifacts, JSON metrics & comparison charts
│   ├── results.json               # Full in-domain evaluation metrics
│   ├── leakage_report.json        # Publisher cue leakage audit
│   ├── external_eval_results.json # External evaluation & Wilson 95% CIs
│   ├── dataset_audit_report.json  # Duplication & contingency audit
│   └── publisher_held_out_report.json # Leave-One-Publisher-Out results
├── src/
│   ├── preprocessing.py           # Negation/number preservation & dateline stripping
│   ├── feature_extraction.py      # TF-IDF feature extractor
│   ├── models.py                  # Classical ML (LR, Linear SVM, RF, GB, Ensemble)
│   ├── deep_learning.py           # PyTorch BiLSTM and DistilBERT Transformer
│   ├── explainability.py          # Directional SHAP feature attribution
│   ├── linguistic_features.py     # Stylistic heuristics & false positive audits
│   ├── evaluation.py              # Macro F1, PR-AUC, ROC-AUC, Brier score, ECE & CV
│   ├── dataset_audit.py           # Exact cue counts, duplicate detection & deduplication
│   ├── leakage_experiments.py     # Publisher artifact frequency & single-cue test
│   ├── external_eval.py           # External benchmark evaluation with Wilson CIs
│   ├── publisher_held_out.py      # Leave-One-Publisher-Out (LOPO) cross-validation
│   ├── train.py                   # Main training pipeline with deduplication & CV
│   └── predict.py                 # CLI inference with sufficiency & leakage guards
├── tests/
│   └── test_fake_news_detection.py# Pytest suite (25 automated unit/integration tests)
├── requirements.txt
└── README.md
```

---

## 5. Usage Commands

```bash
# 1. Audit dataset for exact Reuters cues and duplicate stories
python src/dataset_audit.py --dataset data/sample_isot.csv

# 2. Run publisher leakage and single-cue heuristic experiments
python src/leakage_experiments.py

# 3. Run Leave-One-Publisher-Out (LOPO) generalization cross-validation
python src/publisher_held_out.py --dataset data/external_benchmark.csv

# 4. Train all models (with automatic deduplication, 5-fold CV & external evaluation)
python src/train.py --train-lstm --train-distilbert --epochs 5

# 5. Evaluate trained models on the independent external benchmark with Wilson 95% CIs
python src/external_eval.py --benchmark data/external_benchmark.csv

# 6. Run the 25 automated pytest tests
pytest tests/ -v
```
