import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

from feature_extraction import FeatureExtractor
from preprocessing import TextPreprocessor


def run_publisher_held_out_experiment(
    df,
    text_column='text',
    label_column='label',
    publisher_column='publisher',
    strip_cues=True,
    min_articles_per_publisher=1,
    random_state=42,
):
    """
    Leave-One-Publisher-Out Cross-Validation (LOPO-CV):
    Trains on a subset of publishers and evaluates exclusively on unseen, held-out publishers.
    Compares against a standard Random K-Fold CV baseline where publishers overlap between train and test.
    """
    preprocessor = TextPreprocessor(
        preserve_negation=True,
        preserve_numbers=True,
        strip_publisher_cues=strip_cues,
    )

    df = df.copy()
    df['clean_text'] = df[text_column].apply(preprocessor.preprocess)

    # Filter publishers with enough articles if specified
    pub_counts = df[publisher_column].value_counts()
    valid_pubs = pub_counts[pub_counts >= min_articles_per_publisher].index
    df = df[df[publisher_column].isin(valid_pubs)].reset_index(drop=True)

    publishers = df[publisher_column].values
    unique_publishers = sorted(list(set(publishers)))
    texts = df['clean_text'].values
    labels = df[label_column].values

    # 1. Standard Stratified K-Fold (Random Split - Publishers Overlap)
    n_splits = min(5, len(labels) // 2) if len(labels) >= 4 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    random_cv_accs = []
    random_cv_f1s = []

    for tr_idx, val_idx in skf.split(texts, labels):
        min_df = 1 if len(texts[tr_idx]) < 20 else 2
        vec = FeatureExtractor(max_features=3000, min_df=min_df)
        X_tr = vec.fit_transform(texts[tr_idx])
        X_val = vec.transform(texts[val_idx])

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_tr, labels[tr_idx])
        preds = clf.predict(X_val)

        random_cv_accs.append(accuracy_score(labels[val_idx], preds))
        random_cv_f1s.append(f1_score(labels[val_idx], preds, average='macro', zero_division=0))

    avg_random_acc = float(np.mean(random_cv_accs))
    avg_random_f1 = float(np.mean(random_cv_f1s))

    # 2. Leave-One-Publisher-Out Cross-Validation (LOPO-CV)
    logo = LeaveOneGroupOut()
    lopo_results = []
    all_held_out_preds = []
    all_held_out_trues = []

    for train_idx, test_idx in logo.split(texts, labels, groups=publishers):
        held_out_pub = publishers[test_idx[0]]
        y_test_subset = labels[test_idx]

        # Verify that the held-out publisher has 0 representation in the training set
        train_pubs = set(publishers[train_idx])
        assert held_out_pub not in train_pubs, f"Leakage detected: {held_out_pub} present in training set!"

        # Train model strictly on remaining publishers
        min_df = 1 if len(texts[train_idx]) < 20 else 2
        vec = FeatureExtractor(max_features=3000, min_df=min_df)
        X_train = vec.fit_transform(texts[train_idx])
        X_test = vec.transform(texts[test_idx])

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, labels[train_idx])
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test_subset, y_pred))
        macro_f1 = float(f1_score(y_test_subset, y_pred, average='macro', zero_division=0))

        all_held_out_preds.extend(y_pred)
        all_held_out_trues.extend(y_test_subset)

        lopo_results.append({
            'held_out_publisher': held_out_pub,
            'test_sample_size': len(test_idx),
            'real_count': int((y_test_subset == 0).sum()),
            'fake_count': int((y_test_subset == 1).sum()),
            'held_out_accuracy': round(acc, 4),
            'held_out_macro_f1': round(macro_f1, 4),
        })

    overall_held_out_acc = float(accuracy_score(all_held_out_trues, all_held_out_preds))
    overall_held_out_f1 = float(f1_score(all_held_out_trues, all_held_out_preds, average='macro', zero_division=0))
    generalization_penalty = avg_random_acc - overall_held_out_acc

    return {
        'total_articles': len(df),
        'unique_publishers_count': len(unique_publishers),
        'publishers_evaluated': unique_publishers,
        'standard_random_split_cv': {
            'mean_accuracy': round(avg_random_acc, 4),
            'mean_macro_f1': round(avg_random_f1, 4),
            'folds': n_splits,
        },
        'publisher_held_out_lopo_cv': {
            'overall_accuracy': round(overall_held_out_acc, 4),
            'overall_macro_f1': round(overall_held_out_f1, 4),
            'generalization_drop_due_to_held_out_publisher': round(generalization_penalty, 4),
        },
        'per_held_out_publisher_results': lopo_results,
    }


def plot_publisher_held_out(report, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    per_pub = report['per_held_out_publisher_results']

    names = [r['held_out_publisher'] for r in per_pub]
    accs = [r['held_out_accuracy'] for r in per_pub]
    random_baseline = report['standard_random_split_cv']['mean_accuracy']

    plt.figure(figsize=(11, 5))
    bars = plt.bar(range(len(names)), accs, color='#3ecf8e', alpha=0.85, label='Held-Out Publisher Accuracy')
    plt.axhline(random_baseline, color='#f05c5c', linestyle='--', linewidth=1.5, label=f'Random CV Baseline ({random_baseline:.1%})')

    plt.xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.15)
    plt.title('Leave-One-Publisher-Out Evaluation (Zero Training Overlap)')
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:
        val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.1%}", ha='center', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Leave-One-Publisher-Out (LOPO) cross-validation experiments")
    parser.add_argument('--dataset', type=str, default='data/external_benchmark.csv', help='Path to dataset containing publisher column')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    data_path = os.path.join(project_dir, args.dataset) if not os.path.isabs(args.dataset) else args.dataset

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    pub_col = 'publisher' if 'publisher' in df.columns else ('source' if 'source' in df.columns else None)
    if not pub_col:
        raise ValueError("Dataset must contain a 'publisher' or 'source' column for publisher-held-out experiments.")

    print(f"\n{'=' * 70}")
    print(f"LEAVE-ONE-PUBLISHER-OUT (LOPO) CROSS-VALIDATION")
    print(f"Dataset: {os.path.basename(data_path)} ({len(df)} articles across {df[pub_col].nunique()} publishers)")
    print(f"{'=' * 70}\n")

    report = run_publisher_held_out_experiment(df, publisher_column=pub_col)

    print("--- Overall Generalization Summary ---")
    print(f"Random Split CV Accuracy (Publishers Overlap)   : {report['standard_random_split_cv']['mean_accuracy']:.2%}")
    print(f"Publisher-Held-Out LOPO Accuracy (Zero Overlap) : {report['publisher_held_out_lopo_cv']['overall_accuracy']:.2%}")
    print(f"Cross-Publisher Generalization Gap (Δ)          : {report['publisher_held_out_lopo_cv']['generalization_drop_due_to_held_out_publisher']:.2%}")

    print("\n--- Per-Publisher Held-Out Results ---")
    pub_df = pd.DataFrame(report['per_held_out_publisher_results'])
    print(pub_df[['held_out_publisher', 'test_sample_size', 'held_out_accuracy', 'held_out_macro_f1']].to_string(index=False))

    out_json = os.path.join(models_dir, 'publisher_held_out_report.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    plot_path = os.path.join(models_dir, 'publisher_held_out.png')
    plot_publisher_held_out(report, plot_path)

    print(f"\nReport saved to: {out_json}")
    print(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
