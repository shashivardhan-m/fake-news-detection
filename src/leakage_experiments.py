import argparse
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_extraction import FeatureExtractor
from preprocessing import TextPreprocessor, strip_publisher_artifacts


def analyze_publisher_leakage(df, text_column='text', label_column='label'):
    """
    Examines the frequency of publisher markers (e.g., 'Reuters', datelines)
    between Real (0) and Fake (1) classes to measure dataset leakage.
    """
    reuters_pattern = re.compile(r'\breuters\b', re.IGNORECASE)
    dateline_pattern = re.compile(r'\([A-Za-z\s,/–\-]+\)\s*[-—–]', re.IGNORECASE)

    real_mask = df[label_column] == 0
    fake_mask = df[label_column] == 1

    real_texts = df.loc[real_mask, text_column].astype(str)
    fake_texts = df.loc[fake_mask, text_column].astype(str)

    real_reuters = real_texts.apply(lambda t: bool(reuters_pattern.search(t))).mean()
    fake_reuters = fake_texts.apply(lambda t: bool(reuters_pattern.search(t))).mean()

    real_datelines = real_texts.apply(lambda t: bool(dateline_pattern.search(t))).mean()
    fake_datelines = fake_texts.apply(lambda t: bool(dateline_pattern.search(t))).mean()

    return {
        'total_articles': len(df),
        'real_count': int(real_mask.sum()),
        'fake_count': int(fake_mask.sum()),
        'real_with_reuters_pct': round(float(real_reuters * 100), 2),
        'fake_with_reuters_pct': round(float(fake_reuters * 100), 2),
        'real_with_dateline_pct': round(float(real_datelines * 100), 2),
        'fake_with_dateline_pct': round(float(fake_datelines * 100), 2),
    }


def evaluate_single_cue_heuristic(df, text_column='text', label_column='label', cue='reuters'):
    """
    Evaluates a zero-intelligence heuristic rule:
    Predict 'Real' (0) if cue exists in text, else 'Fake' (1).
    Demonstrates how much performance comes from dataset artifact shortcuts.
    """
    pattern = re.compile(rf'\b{re.escape(cue)}\b', re.IGNORECASE)
    y_true = df[label_column].values
    # If cue is present -> 0 (Real), else 1 (Fake)
    y_pred = df[text_column].astype(str).apply(lambda t: 0 if pattern.search(t) else 1).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        'cue': cue,
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'macro_f1': round(float(macro_f1), 4),
        'confusion_matrix': cm,
    }


def run_leakage_ablation_study(train_df, external_df=None, test_size=0.2, random_state=42):
    """
    Compares two models:
      1. Model A (Biased): Trained on raw text with publisher cues preserved.
      2. Model B (Debiased): Trained on cleaned text with publisher cues stripped.
    Evaluates both on:
      - In-domain test set with cues
      - In-domain test set stripped of cues
      - External independent dataset (unseen publishers)
    """
    preprocessor_raw = TextPreprocessor(preserve_negation=True, preserve_numbers=True, strip_publisher_cues=False)
    preprocessor_stripped = TextPreprocessor(preserve_negation=True, preserve_numbers=True, strip_publisher_cues=True)

    # Prepare texts
    raw_texts = train_df['text'].astype(str)
    labels = train_df['label'].values

    train_raw, test_raw, y_train, y_test = train_test_split(
        raw_texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Model A: Trained on raw text
    X_train_raw_clean = [preprocessor_raw.preprocess(t) for t in train_raw]
    X_test_raw_clean = [preprocessor_raw.preprocess(t) for t in test_raw]
    X_test_stripped_clean = [preprocessor_stripped.preprocess(t) for t in test_raw]

    vec_a = FeatureExtractor(max_features=5000)
    X_tr_a = vec_a.fit_transform(X_train_raw_clean)
    X_te_a_raw = vec_a.transform(X_test_raw_clean)
    X_te_a_stripped = vec_a.transform(X_test_stripped_clean)

    model_a = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_a.fit(X_tr_a, y_train)

    # Model B: Trained on stripped text
    X_train_stripped_clean = [preprocessor_stripped.preprocess(t) for t in train_raw]

    vec_b = FeatureExtractor(max_features=5000)
    X_tr_b = vec_b.fit_transform(X_train_stripped_clean)
    X_te_b_raw = vec_b.transform(X_test_raw_clean)
    X_te_b_stripped = vec_b.transform(X_test_stripped_clean)

    model_b = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_b.fit(X_tr_b, y_train)

    def eval_model(model, X, y):
        preds = model.predict(X)
        return {
            'accuracy': round(float(accuracy_score(y, preds)), 4),
            'macro_f1': round(float(f1_score(y, preds, average='macro', zero_division=0)), 4),
            'f1': round(float(f1_score(y, preds, zero_division=0)), 4),
        }

    results = {
        'model_a_raw_trained': {
            'eval_on_in_domain_raw': eval_model(model_a, X_te_a_raw, y_test),
            'eval_on_in_domain_stripped': eval_model(model_a, X_te_a_stripped, y_test),
        },
        'model_b_stripped_trained': {
            'eval_on_in_domain_raw': eval_model(model_b, X_te_b_raw, y_test),
            'eval_on_in_domain_stripped': eval_model(model_b, X_te_b_stripped, y_test),
        }
    }

    # External dataset evaluation
    if external_df is not None and len(external_df) > 0:
        ext_texts = external_df['text'].astype(str)
        ext_labels = external_df['label'].values

        ext_raw_clean = [preprocessor_raw.preprocess(t) for t in ext_texts]
        ext_stripped_clean = [preprocessor_stripped.preprocess(t) for t in ext_texts]

        X_ext_a = vec_a.transform(ext_raw_clean)
        X_ext_b = vec_b.transform(ext_stripped_clean)

        results['model_a_raw_trained']['eval_on_external'] = eval_model(model_a, X_ext_a, ext_labels)
        results['model_b_stripped_trained']['eval_on_external'] = eval_model(model_b, X_ext_b, ext_labels)

    return results


def plot_leakage_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    categories = ['In-Domain (Raw)', 'In-Domain (Stripped)']
    has_ext = 'eval_on_external' in results['model_a_raw_trained']
    if has_ext:
        categories.append('External Benchmark')

    acc_a = [
        results['model_a_raw_trained']['eval_on_in_domain_raw']['accuracy'],
        results['model_a_raw_trained']['eval_on_in_domain_stripped']['accuracy'],
    ]
    acc_b = [
        results['model_b_stripped_trained']['eval_on_in_domain_raw']['accuracy'],
        results['model_b_stripped_trained']['eval_on_in_domain_stripped']['accuracy'],
    ]

    if has_ext:
        acc_a.append(results['model_a_raw_trained']['eval_on_external']['accuracy'])
        acc_b.append(results['model_b_stripped_trained']['eval_on_external']['accuracy'])

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, acc_a, width, label='Model A (Trained with Cues / Overfitted)', color='#f05c5c')
    plt.bar(x + width/2, acc_b, width, label='Model B (Trained with Cues Stripped)', color='#3ecf8e')

    plt.ylabel('Accuracy')
    plt.title('Impact of Publisher Artifact Leakage on Generalization')
    plt.xticks(x, categories)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(categories)):
        plt.text(x[i] - width/2, acc_a[i] + 0.02, f"{acc_a[i]:.2%}", ha='center', fontsize=9)
        plt.text(x[i] + width/2, acc_b[i] + 0.02, f"{acc_b[i]:.2%}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run publisher & source leakage experiments on fake news datasets")
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset CSV (defaults to ISOT True/Fake or data/sample_isot.csv)')
    parser.add_argument('--external', type=str, default='data/external_benchmark.csv', help='Path to external benchmark CSV')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Load dataset
    if args.dataset and os.path.exists(args.dataset):
        df = pd.read_csv(args.dataset)
    else:
        true_path = os.path.join(project_dir, 'True.csv')
        fake_path = os.path.join(project_dir, 'Fake.csv')
        if os.path.exists(true_path) and os.path.exists(fake_path):
            t_df = pd.read_csv(true_path)
            f_df = pd.read_csv(fake_path)
            t_df['label'] = 0
            f_df['label'] = 1
            df = pd.concat([t_df, f_df], ignore_index=True)
            df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
        elif os.path.exists(os.path.join(project_dir, 'data', 'sample_isot.csv')):
            df = pd.read_csv(os.path.join(project_dir, 'data', 'sample_isot.csv'))
            df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
        else:
            raise FileNotFoundError("Could not find True.csv/Fake.csv or data/sample_isot.csv.")

    print("\n" + "=" * 60)
    print("STEP 1: PUBLISHER CUE & ARTIFACT FREQUENCY ANALYSIS")
    print("=" * 60)
    leakage_stats = analyze_publisher_leakage(df)
    for k, v in leakage_stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("STEP 2: SINGLE-CUE 'REUTERS' HEURISTIC EVALUATION")
    print("=" * 60)
    heuristic_results = evaluate_single_cue_heuristic(df, cue='reuters')
    for k, v in heuristic_results.items():
        print(f"  {k}: {v}")

    external_df = None
    ext_path = os.path.join(project_dir, args.external) if not os.path.isabs(args.external) else args.external
    if os.path.exists(ext_path):
        external_df = pd.read_csv(ext_path)
        external_df['text'] = external_df['title'].astype(str) + ' ' + external_df['text'].astype(str)

    print("\n" + "=" * 60)
    print("STEP 3: ABLATION STUDY: CUE-TRAINED VS DEBIASED MODEL")
    print("=" * 60)
    ablation_results = run_leakage_ablation_study(df, external_df=external_df)
    print(json.dumps(ablation_results, indent=2))

    # Save full leakage report
    full_report = {
        'leakage_stats': leakage_stats,
        'heuristic_baseline': heuristic_results,
        'ablation_study': ablation_results,
    }
    report_path = os.path.join(models_dir, 'leakage_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)

    plot_path = os.path.join(models_dir, 'leakage_comparison.png')
    plot_leakage_results(ablation_results, plot_path)

    print(f"\nLeakage analysis completed successfully!")
    print(f"Report saved to: {report_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
