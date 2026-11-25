# Independent External Benchmark: Methodology & Provenance

This document details the curation methodology, verification standards, and contamination checks for `data/external_benchmark.csv`.

---

## 1. Motivation: Overcoming Single-Dataset Confounders

Standard fake news benchmarks such as the ISOT dataset suffer from a severe structural confounder:
- All real articles were scraped from a single outlet (`Reuters.com`).
- All fake articles were gathered from uncurated unreliable websites.

Consequently, models trained on ISOT frequently achieve $>99\%$ accuracy not by identifying false factual assertions, but by recognizing publisher-specific dateline syntax (e.g. `WASHINGTON (Reuters) -`) and journalistic tone. 

To evaluate whether a trained model generalizes to genuinely unseen sources, this benchmark provides an independent evaluation set gathered from distinct publishers across diverse domains.

---

## 2. Labelling Protocol & Ground Truth Verification

### Principle: Labels Are Evidence-Based, Not Publisher-Based
Articles are **not** labeled as Real simply because they originate from mainstream media, nor are they labeled as Fake simply because of the outlet. Every ground-truth label in this benchmark is anchored in third-party, publicly verifiable evidence:

- **Verified Real Articles (`label: 0`)**:
  - Sourced from documented reports published by wire services and journalistic organizations (Associated Press, BBC News, NPR, The Guardian, Nature News, Wall Street Journal, ProPublica).
  - Each item corresponds to an official primary record (e.g. peer-reviewed journal papers, government regulatory filings, official central bank rate decisions, EPA bulletins).
  - Articles have had wire datelines and agency signatures stripped to ensure fair textual comparison.

- **Verified Fake / Misinformation Articles (`label: 1`)**:
  - Sourced from viral claims, conspiracy outlets, pseudoscientific newsletters, and satirical publications.
  - Every single item has an explicit, documented debunking URL from an **International Fact-Checking Network (IFCN) signatory**:
    - **PolitiFact** (e.g. "Pants on Fire" rulings on secret digital dollar expiry legislation)
    - **Snopes** (e.g. weight-loss scams, Antarctic pyramid hoaxes)
    - **FactCheck.org** (e.g. alkaline water claims, garlic cure myths)
    - **Reuters Fact Check** (e.g. telecommunications brainwave radiation hoaxes)
  - Subtly written falsehoods (formal tone without sensationalist punctuation) are included to prevent the model from relying merely on exclamation mark heuristics.

---

## 3. Schema & Fields

| Field Name | Type | Description |
|---|---|---|
| `id` | string | Unique benchmark identifier (e.g. `EXT-001`) |
| `publisher` | string | Reported publisher or outlet |
| `title` | string | Full headline |
| `text` | string | Full article body / claim text |
| `date` | string (ISO) | Original publication date (`YYYY-MM-DD`) |
| `label` | integer | Ground truth: `0` (Real / Verified True) or `1` (Fake / Debunked False) |
| `factuality_verdict` | string | Granular verdict: `true`, `mostly-true`, `false`, `pants-on-fire`, `satire` |
| `verification_url` | string (URL) | Direct link to the primary fact-checker ruling or verified primary document |
| `fact_checking_notes` | string | Detailed explanation of evidence and reason for verdict |
| `domain_category` | string | Subject domain (`science`, `economy`, `health`, `politics`, `environment`, `technology`, `society`) |

---

## 4. Contamination & Deduplication Checks

To ensure that in-domain training data does not contaminate the benchmark:
1. **Zero Text Overlap**: All benchmark articles were cross-referenced against ISOT articles (`True.csv` and `Fake.csv`) using both exact MD5 string matching and TF-IDF cosine similarity.
2. **Temporal Separation**: Benchmark articles date from 2022–2023, whereas ISOT articles date from 2016–2017.
3. **No Dateline Markers**: Real articles in this benchmark do not contain `(Reuters)` or news-agency lead markers, testing whether the model can classify journalistic prose without dataset-specific shortcuts.
