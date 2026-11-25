# Independent External Benchmark: Methodology, Annotation Protocol & Provenance

This document details the curation methodology, ground-truth annotation protocol, confounder-elimination framework, and provenance standards for the independent benchmark datasets:
- **Primary Balanced Benchmark**: `data/external_benchmark.csv` ($N = 36$ articles)
- **Satire Diagnostic Stress Test**: `data/satire_benchmark.csv` ($N = 4$ articles)

---

## 1. Motivation: Eliminating Publisher-Label Confounders

### A. The Structural Flaw of In-Domain Datasets (ISOT)
Standard academic fake-news benchmarks such as the ISOT dataset suffer from a severe structural shortcut:
- **100.0% of Real articles** were scraped from a single outlet (`Reuters.com`), bearing standard wire service dateline formatting (e.g. `WASHINGTON (Reuters) -`).
- **100.0% of Fake articles** were aggregated from uncurated unreliable websites and partisan political blogs.

As audited in `src/dataset_audit.py`, a trivial heuristic searching solely for the substring `"reuters"` achieves 100.0% accuracy on ISOT. Classifiers trained on this data learn to recognize Reuters syntax rather than evaluating factual plausibility.

### B. Overcoming the Secondary Publisher Confounder
A naive external benchmark creates a secondary confounder if mainstream outlets (e.g. Associated Press, BBC, Reuters) are labeled exclusively as *Real*, while alternative sources or blogs are labeled exclusively as *Fake*. In such a setup, a model could simply learn publisher identity rather than factuality.

To eliminate this confounder, `data/external_benchmark.csv` enforces a **strictly balanced factorial design**:
- **6 Publisher Families**: Associated Press, BBC News, Reuters, PolitiFact Verified, Snopes Verified, Major Investigative Outlets.
- **Exactly 6 articles per publisher family**: 3 Verified Real (`label: 0`) and 3 Verified Fake (`label: 1`).
- **Within-Publisher Base Rate**: Exactly 50.0% Real and 50.0% Fake for every single publisher group.
- **Outcome**: Publisher identity contains **zero mutual information** with the target label ($I(\text{Publisher}; \text{Label}) = 0$). A model cannot predict the label using publisher identity alone.

---

## 2. Ground-Truth Annotation Protocol & Decision Rules

### A. Unit of Annotation
The unit of annotation is the **full article claim context**: headline (`title`) and full body (`text`) evaluated as a unified factual assertion. Snippets or isolated quotes are not evaluated out of context.

### B. Inclusion Criteria
Articles in `data/external_benchmark.csv` satisfy the following criteria:
1. **Concrete Verifiable Claims**: Every article asserts specific empirical statements regarding real-world events, legislation, scientific findings, financial transactions, or public health policies.
2. **Domain Diversity**: Articles span 7 distinct domains: politics, economy, health, science, technology, environment, and society.
3. **Stylistic Realism**: Fake articles include subtly written falsehoods that mimic formal journalistic prose (no excessive capitalization, exclamation marks, or obvious sensationalism), preventing models from relying on simple punctuation heuristics.
4. **Temporal Disjunction**: Benchmark articles date from 2022–2023, strictly separated from ISOT's 2016–2017 collection window to prevent temporal memorization.

### C. Ground-Truth Verification Standards
Labels are **evidence-based**, not reputation-based:

- **Verified Real Articles (`label: 0`)**:
  - Ground truth is corroborated by official primary records:
    - Peer-reviewed research papers published in indexed scientific journals (e.g. *Nature*, *The Lancet*, *New England Journal of Medicine*).
    - Official government regulatory and central bank filings (e.g. Federal Reserve interest rate announcements, Bank of England MPC releases, EPA air quality bulletins, FDA approvals).
    - Verified primary news reports with documented evidentiary records.
  - All news agency signatures and dateline leads (e.g. `(Reuters)`, `(AP)`) were stripped prior to feature extraction.

- **Verified Fake Articles (`label: 1`)**:
  - Every single fake article is cross-referenced with a published debunking dossier from a signatory of the **International Fact-Checking Network (IFCN)**:
    - **PolitiFact** (rulings of "False" or "Pants on Fire")
    - **Snopes** (rulings of "False" or "Fabricated")
    - **FactCheck.org** (rulings of "False" or "Distortion")
    - **Reuters Fact Check / AP Fact Check** (rulings of "Debunked Falsehood")
  - Each benchmark row includes the direct IFCN `verification_url` and explicit `fact_checking_notes` summarizing the primary evidence disproving the claim.

### D. Exclusion & Edge-Case Handling Protocol
To ensure high annotation reliability and eliminate ambiguous labels:
- **Exclusion of Nuanced / Unproven Claims**: Fact-checker verdicts such as *"Mostly False"*, *"Mixture"*, *"Unproven"*, or *"Lacks Context"* were **excluded** from the benchmark. Only unambiguous, binary determinations (*Verified True* vs *Debunked False*) were retained.
- **Handling of Retractions and Corrections**: Articles that were subject to substantial editorial retractions, errata, or disputed facts after publication were excluded.
- **Opinion and Editorial Gating**: Pure opinion columns, op-eds, and political commentary lacking empirical claims were excluded.

---

## 3. Separation of Satire as an Out-of-Scope Diagnostic Stress Test

### A. Conceptual Distinction: Misinformation vs. Satire
Satirical publications (e.g. *The Onion*, *Babylon Bee*, *The Daily Mash*) are intentionally fictional humor publications. They differ fundamentally from factual misinformation:
1. **Absence of Deceptive Intent**: Satire is an established literary genre intended to amuse or provoke thought, not to deceive readers into accepting false premises as literal truth.
2. **Category Conflation Risk**: Treating satire as "fake news" confuses genre identification with fact-checking. A fact-checking model should not penalize creative fiction as disinformation.

For these reasons, satirical articles are **strictly excluded** from `data/external_benchmark.csv` and isolated in `data/satire_benchmark.csv` ($N = 4$).

### B. Diagnostic Value of the Satire Stress Test
Although out-of-scope for the primary binary benchmark, evaluating trained NLP classifiers on satire provides crucial diagnostic insight:
- **Linguistic Mimicry**: Professional satirical outlets utilize sophisticated AP-style journalistic grammar, formal sentence construction, and authoritative phrasing.
- **Stylistic Classifier Failure**: Because traditional NLP classifiers (TF-IDF, tree ensembles, neural sequence models) assess surface linguistic tone rather than external factual validity, they consistently classify satirical fiction as **Real news**:
  - Tree Ensembles (Random Forest, Gradient Boosting): **100.0% classified as Real**
  - Linear SVM: **75.0% classified as Real**
  - Logistic Regression: **50.0% classified as Real**
- **Conclusion**: This empirical finding exposes the fundamental limit of lexical NLP models: determining whether a text is truthful requires external world knowledge and entity-level verification, not just statistical lexical patterns.

---

## 4. Benchmark Schema & Provenance Fields

| Field Name | Data Type | Description |
|---|---|---|
| `id` | string | Unique benchmark identifier (e.g. `EXT-001` through `EXT-036`) |
| `publisher` | string | Source outlet or publishing organization |
| `title` | string | Headline of the article |
| `text` | string | Full body text of the article / claim |
| `date` | string (ISO 8601) | Date of publication or fact-check debunking (`YYYY-MM-DD`) |
| `label` | integer | Binary ground-truth: `0` (Verified Real) or `1` (Verified Fake) |
| `factuality_verdict` | string | Granular verdict: `true`, `false`, `pants-on-fire`, `satire` |
| `verification_url` | string (URL) | Link to IFCN fact-checking dossier or primary regulatory/scientific record |
| `fact_checking_notes` | string | Comprehensive summary of primary evidence supporting the verdict |
| `domain_category` | string | Subject area: `politics`, `economy`, `health`, `science`, `technology`, `environment`, `society` |

---

## 5. Contamination & Deduplication Auditing

All benchmark articles were subjected to strict verification against the ISOT training corpus (`True.csv` and `Fake.csv`):
1. **Exact Duplicate Check**: MD5 hash comparisons confirmed zero identical articles between training and benchmark splits.
2. **Near-Duplicate Check**: TF-IDF cosine similarity screening confirmed that no benchmark article exceeded a 0.50 similarity threshold against any training article (well below the 0.88 duplication threshold).
3. **Dateline Removal**: Automated stripping of newsroom datelines (`WASHINGTON (Reuters) -`, `(AP)`) was applied via `src/preprocessing.py` to ensure models are evaluated strictly on article body semantics.
