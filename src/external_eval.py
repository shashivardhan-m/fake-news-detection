import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation import ModelEvaluator
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from preprocessing import TextPreprocessor


def evaluate_on_external_dataset(
    model_name,
    model,
    extractor,
    external_df,
    preprocessor,
    text_column='text',
    label_column='label',
    source_column='source',
):
    """
    Evaluates a single model on an unseen external dataset,
    providing overall metrics and source-level breakdown.
    """
    texts = external_df[text_column].astype(str).tolist()
    y_true = external_df[label_column].values

    cleaned_texts = [preprocessor.preprocess(t) for t in texts]
    features = extractor.transform(cleaned_texts)

    y_pred = model.predict(features)
    y_proba = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else None

    evaluator = ModelEvaluator(y_true, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    # Per-source breakdown if source column is present
    source_breakdown = {}
    if source_column in external_df.columns:
        for src in external_df[source_column].unique():
            idx = external_df[source_column] == src
            sub_true = y_true[idx]
            sub_pred = y_pred[idx]
            source_breakdown[src] = {
                'count': int(idx.sum()),
                'accuracy': round(float(np.mean(sub_true == sub_pred)), 4),
                'ground_truth': 'Fake' if sub_true[0] == 1 else 'Real',
            }

    return {
        'metrics': metrics,
        'source_breakdown': source_breakdown,
    }


def run_full_external_evaluation(project_dir, external_csv_path, strip_cues=True):
    models_dir = os.path.join(project_dir, 'models')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    results_path = os.path.join(models_dir, 'results.json')

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}. Train models first.")

    extractor = FeatureExtractor()
    extractor.load(vectorizer_path)

    preprocessor = TextPreprocessor(
        preserve_negation=True,
        preserve_numbers=True,
        strip_publisher_cues=strip_cues,
    )

    external_df = pd.read_csv(external_csv_path)
    if 'text' not in external_df.columns:
        if 'title' in external_df.columns:
            external_df['text'] = external_df['title'].astype(str)
        else:
            raise ValueError("External CSV must contain a 'text' column.")
    elif 'title' in external_df.columns:
        external_df['text'] = external_df['title'].astype(str) + ' ' + external_df['text'].astype(str)

    in_domain_results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            in_domain_results = json.load(f)

    evaluation_report = {}
    comparison_table = []

    model_keys = list(MODEL_CONFIGS.keys()) + ['ensemble']
    for model_type in model_keys:
        filename = FakeNewsModel.get_filename(model_type)
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            continue

        display_name = FakeNewsModel.get_display_name(model_type)
        model = FakeNewsModel(model_type=model_type)
        model.load(model_path)

        res = evaluate_on_external_dataset(
            display_name, model, extractor, external_df, preprocessor,
        )
        evaluation_report[display_name] = res

        ext_acc = res['metrics']['accuracy']
        ext_macro_f1 = res['metrics']['macro_f1']

        in_acc = in_domain_results.get(display_name, {}).get('accuracy', np.nan)
        in_macro_f1 = in_domain_results.get(display_name, {}).get('macro_f1', in_domain_results.get(display_name, {}).get('f1', np.nan))

        acc_drop = in_acc - ext_acc if not np.isnan(in_acc) else np.nan
        f1_drop = in_macro_f1 - ext_macro_f1 if not np.isnan(in_macro_f1) else np.nan

        comparison_table.append({
            'Model': display_name,
            'In-Domain Acc': in_acc,
            'External Acc': ext_acc,
            'Accuracy Drop (Δ)': acc_drop,
            'In-Domain Macro F1': in_macro_f1,
            'External Macro F1': ext_macro_f1,
            'Macro F1 Drop (Δ)': f1_drop,
        })

    summary_df = pd.DataFrame(comparison_table)
    return evaluation_report, summary_df


def plot_generalization_drop(summary_df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if summary_df.empty:
        return

    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    in_acc = summary_df['In-Domain Acc'].fillna(0)
    ext_acc = summary_df['External Acc'].fillna(0)

    plt.bar(x - width/2, in_acc, width, label='In-Domain Test (ISOT)', color='#4a9eff')
    plt.bar(x + width/2, ext_acc, width, label='External Benchmark (Unseen)', color='#e8a838')

    plt.xticks(x, summary_df['Model'], rotation=25, ha='right')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.title('Generalization Gap: In-Domain Test vs Independent External Benchmark')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(summary_df)):
        plt.text(x[i] - width/2, in_acc.iloc[i] + 0.02, f"{in_acc.iloc[i]:.2%}", ha='center', fontsize=8)
        plt.text(x[i] + width/2, ext_acc.iloc[i] + 0.02, f"{ext_acc.iloc[i]:.2%}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on independent external dataset")
    parser.add_argument('--benchmark', type=str, default='data/external_benchmark.csv', help='Path to external dataset CSV')
    parser.add_argument('--keep-cues', action='store_true', help='Do not strip publisher cues during evaluation')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    benchmark_path = os.path.join(project_dir, args.benchmark) if not os.path.isabs(args.benchmark) else args.benchmark

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"External benchmark not found at {benchmark_path}")

    print(f"\n{'=' * 65}")
    print(f"EVALUATING MODELS ON INDEPENDENT EXTERNAL BENCHMARK")
    print(f"Benchmark file: {benchmark_path}")
    print(f"{'=' * 65}\n")

    report, summary_df = run_full_external_evaluation(
        project_dir, benchmark_path, strip_cues=not args.keep_cues
    )

    print("--- Generalization Gap Summary ---")
    print(summary_df.to_string(index=False))

    # Save outputs
    out_json = os.path.join(models_dir, 'external_eval_results.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    plot_path = os.path.join(models_dir, 'external_generalization.png')
    plot_generalization_drop(summary_df, plot_path)

    print(f"\nExternal evaluation complete!")
    print(f"Full metrics saved to: {out_json}")
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
