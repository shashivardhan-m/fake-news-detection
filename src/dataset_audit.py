import argparse
import difflib
import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def audit_reuters_cues_exact(df, text_column='text', label_column='label'):
    """
    Directly audits exact Reuters cue counts and percentages on the dataset.
    Reports full contingency table and performance of the heuristic rule.
    """
    dateline_patterns = [
        re.compile(r'^[A-Z\s,/–\-]+(?:\([A-Za-z\s,/–\-]+\))?\s*\(Reuters\)\s*[-—–]', re.IGNORECASE),
        re.compile(r'^\(Reuters\)\s*[-—–]', re.IGNORECASE),
        re.compile(r'\b(?:WASHINGTON|LONDON|PARIS|MOSCOW|BERLIN|GENEVA|BEIJING|NEW YORK)\s+\(Reuters\)\b', re.IGNORECASE),
    ]
    reuters_word_pattern = re.compile(r'\breuters\b', re.IGNORECASE)

    real_df = df[df[label_column] == 0]
    fake_df = df[df[label_column] == 1]

    n_real = len(real_df)
    n_fake = len(fake_df)
    total = len(df)

    real_dateline_matches = 0
    real_word_matches = 0
    fake_dateline_matches = 0
    fake_word_matches = 0

    for text in real_df[text_column].astype(str):
        if any(p.search(text) for p in dateline_patterns):
            real_dateline_matches += 1
        if reuters_word_pattern.search(text):
            real_word_matches += 1

    for text in fake_df[text_column].astype(str):
        if any(p.search(text) for p in dateline_patterns):
            fake_dateline_matches += 1
        if reuters_word_pattern.search(text):
            fake_word_matches += 1

    # Contingency matrix for "reuters in text" heuristic predicting Real (0)
    # y_true: Real=0, Fake=1
    # heuristic predicts Real (0) if reuters in text, Fake (1) otherwise
    tp_real = real_word_matches                 # Correctly identified Real
    fn_real = n_real - real_word_matches        # Real predicted as Fake
    fp_real = fake_word_matches                 # Fake predicted as Real
    tn_real = n_fake - fake_word_matches        # Correctly identified Fake

    heuristic_accuracy = (tp_real + tn_real) / total if total > 0 else 0.0

    return {
        'total_articles': total,
        'real_count': n_real,
        'fake_count': n_fake,
        'real_with_exact_dateline': {
            'count': real_dateline_matches,
            'percentage': round((real_dateline_matches / n_real * 100), 2) if n_real else 0.0,
        },
        'real_with_reuters_word': {
            'count': real_word_matches,
            'percentage': round((real_word_matches / n_real * 100), 2) if n_real else 0.0,
        },
        'fake_with_exact_dateline': {
            'count': fake_dateline_matches,
            'percentage': round((fake_dateline_matches / n_fake * 100), 2) if n_fake else 0.0,
        },
        'fake_with_reuters_word': {
            'count': fake_word_matches,
            'percentage': round((fake_word_matches / n_fake * 100), 2) if n_fake else 0.0,
        },
        'heuristic_contingency_table': {
            'true_real_predicted_real (TP)': tp_real,
            'true_real_predicted_fake (FN)': fn_real,
            'true_fake_predicted_real (FP)': fp_real,
            'true_fake_predicted_fake (TN)': tn_real,
        },
        'heuristic_accuracy': round(float(heuristic_accuracy), 4),
        'heuristic_sensitivity_real': round(float(tp_real / n_real), 4) if n_real else 0.0,
        'heuristic_specificity_fake': round(float(tn_real / n_fake), 4) if n_fake else 0.0,
    }


def audit_duplicates_and_contamination(df, text_column='text', label_column='label', near_dup_threshold=0.88):
    """
    Audits exact duplicates and near-duplicates (syndicated stories)
    to verify that in-domain accuracy is not inflated by duplicated articles.
    """
    total = len(df)
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values

    # 1. Exact duplicates
    exact_duplicate_mask = df.duplicated(subset=[text_column], keep='first')
    exact_duplicate_count = int(exact_duplicate_mask.sum())

    # Contradictory duplicates: exact same text with different labels
    cross_label_duplicates = 0
    grouped = df.groupby(text_column)[label_column].nunique()
    cross_label_duplicates = int((grouped > 1).sum())

    # 2. Near duplicates using TF-IDF cosine similarity
    vec = TfidfVectorizer(max_features=2000, stop_words='english')
    X_tfidf = vec.fit_transform(texts)
    sim_matrix = cosine_similarity(X_tfidf)

    np.fill_diagonal(sim_matrix, 0.0)
    near_dup_pairs = []

    rows, cols = np.where(sim_matrix >= near_dup_threshold)
    seen = set()
    for r, c in zip(rows, cols):
        if r < c:
            pair_key = (r, c)
            if pair_key not in seen:
                seen.add(pair_key)
                near_dup_pairs.append({
                    'index_1': int(r),
                    'index_2': int(c),
                    'similarity': round(float(sim_matrix[r, c]), 4),
                    'same_label': bool(labels[r] == labels[c]),
                    'label_1': int(labels[r]),
                    'label_2': int(labels[c]),
                })

    return {
        'total_rows': total,
        'exact_duplicates_count': exact_duplicate_count,
        'exact_duplicate_rate': round(exact_duplicate_count / total * 100, 2) if total else 0.0,
        'cross_label_contradiction_count': cross_label_duplicates,
        'near_duplicate_pairs_found': len(near_dup_pairs),
        'near_duplicate_threshold': near_dup_threshold,
        'sample_near_duplicate_pairs': near_dup_pairs[:5],
    }


def clean_and_deduplicate_dataset(df, text_column='text', label_column='label', drop_near_duplicates=True, threshold=0.90):
    """
    Produces a strictly deduplicated dataset with zero exact duplicates
    and removed near-duplicate pairs to prevent test contamination.
    """
    initial_len = len(df)
    cleaned_df = df.drop_duplicates(subset=[text_column], keep='first').copy()
    exact_dropped = initial_len - len(cleaned_df)

    if drop_near_duplicates and len(cleaned_df) > 1:
        texts = cleaned_df[text_column].astype(str).tolist()
        vec = TfidfVectorizer(max_features=2000, stop_words='english')
        X_tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(X_tfidf)
        np.fill_diagonal(sim_matrix, 0.0)

        to_drop = set()
        rows, cols = np.where(sim_matrix >= threshold)
        for r, c in zip(rows, cols):
            if r < c and r not in to_drop and c not in to_drop:
                to_drop.add(c)

        if to_drop:
            cleaned_df = cleaned_df.iloc[[i for i in range(len(cleaned_df)) if i not in to_drop]].copy()

    final_len = len(cleaned_df)
    return cleaned_df, {
        'initial_count': initial_len,
        'final_count': final_len,
        'removed_count': initial_len - final_len,
        'exact_duplicates_removed': exact_dropped,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit dataset for publisher cue leakage, duplicates, and contamination")
    parser.add_argument('--dataset', type=str, default='data/sample_isot.csv', help='Path to dataset CSV')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, args.dataset) if not os.path.isabs(args.dataset) else args.dataset

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

    print(f"\n{'=' * 65}")
    print(f"AUDITING DATASET: {os.path.basename(data_path)} ({len(df)} articles)")
    print(f"{'=' * 65}")

    print("\n--- 1. Exact Reuters Cue Frequency Audit ---")
    cues_report = audit_reuters_cues_exact(df)
    print(json.dumps(cues_report, indent=2))

    print("\n--- 2. Duplicate & Near-Duplicate Contamination Audit ---")
    dup_report = audit_duplicates_and_contamination(df)
    print(json.dumps(dup_report, indent=2))

    out_file = os.path.join(project_dir, 'models', 'dataset_audit_report.json')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({'cues_audit': cues_report, 'duplicate_audit': dup_report}, f, indent=2)

    print(f"\nAudit complete! Saved detailed report to: {out_file}")


if __name__ == '__main__':
    main()
