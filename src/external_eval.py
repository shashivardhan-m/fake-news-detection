import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

from evaluation import ModelEvaluator
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from preprocessing import TextPreprocessor


def wilson_score_interval(successes, total, confidence=0.95):
    """
    Computes Wilson score interval for binomial proportions.
    Provides robust confidence bounds especially on small sample sizes (e.g. N=36 or N=6).
    """
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def bootstrap_metric_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence=0.95, seed=42):
    """
    Computes non-parametric bootstrap confidence interval for a metric.
    """
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_func(y_true[idx], y_pred[idx]))

    if not scores:
        return 0.0, 1.0

    lower = float(np.percentile(scores, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(scores, (1 + confidence) / 2 * 100))
    return round(lower, 4), round(upper, 4)


def evaluate_on_external_dataset(
    model_name,
    model,
    model_type,
    extractor,
    external_df,
    preprocessor,
    text_column='text',
    label_column='label',
    source_column='publisher',
):
    """
    Evaluates a single model on an unseen external benchmark with:
    - Wilson score confidence intervals for accuracy
    - Non-parametric bootstrap intervals for Macro F1
    - Per-publisher breakdown
    """
    texts = external_df[text_column].astype(str).tolist()
    y_true = external_df[label_column].values

    cleaned_texts = [preprocessor.preprocess(t) for t in texts]

    if model_type == 'deep_learning':
        y_pred = model.predict(cleaned_texts)
        y_proba = model.predict_proba(cleaned_texts)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        features = extractor.transform(cleaned_texts)
        y_pred = model.predict(features)
        y_proba = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else None

    evaluator = ModelEvaluator(y_true, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    acc = metrics['accuracy']
    n_total = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))

    wilson_low, wilson_high = wilson_score_interval(n_correct, n_total)
    macro_f1_func = lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
    boot_f1_low, boot_f1_high = bootstrap_metric_ci(y_true, y_pred, macro_f1_func)

    metrics['confidence_intervals'] = {
        'accuracy_wilson_95ci': [round(wilson_low, 4), round(wilson_high, 4)],
        'macro_f1_bootstrap_95ci': [boot_f1_low, boot_f1_high],
        'sample_size': n_total,
        'correct_predictions': n_correct,
    }

    # Per-publisher breakdown
    publisher_col = source_column if source_column in external_df.columns else ('source' if 'source' in external_df.columns else None)
    source_breakdown = {}

    if publisher_col:
        for pub in sorted(external_df[publisher_col].unique()):
            idx = external_df[publisher_col] == pub
            sub_true = y_true[idx]
            sub_pred = y_pred[idx]
            sub_n = int(idx.sum())
            sub_correct = int(np.sum(sub_true == sub_pred))
            sub_acc = float(sub_correct / sub_n) if sub_n else 0.0
            s_low, s_high = wilson_score_interval(sub_correct, sub_n)

            source_breakdown[pub] = {
                'sample_size': sub_n,
                'real_count': int((sub_true == 0).sum()),
                'fake_count': int((sub_true == 1).sum()),
                'predicted_real': int((sub_pred == 0).sum()),
                'predicted_fake': int((sub_pred == 1).sum()),
                'correct_count': sub_correct,
                'accuracy': round(sub_acc, 4),
                'wilson_95ci': [round(s_low, 4), round(s_high, 4)],
                'macro_f1': round(float(f1_score(sub_true, sub_pred, average='macro', zero_division=0)), 4),
            }

    return {
        'metrics': metrics,
        'source_breakdown': source_breakdown,
    }


def evaluate_satire_benchmark(
    models_dict,
    satire_df,
    preprocessor,
    extractor,
    text_column='text',
):
    """
    Evaluates trained models on an isolated satirical dataset (e.g. The Onion, Babylon Bee, The Daily Mash).
    Reports how often models misclassify satirical fiction as legitimate Real news.
    Satire is treated as an out-of-scope diagnostic stress test rather than factual misinformation.
    """
    texts = satire_df[text_column].astype(str).tolist()
    cleaned_texts = [preprocessor.preprocess(t) for t in texts]
    features = extractor.transform(cleaned_texts)

    results = {}
    for name, item in models_dict.items():
        model = item['model']
        m_type = item['type']

        if m_type == 'deep_learning':
            preds = model.predict(cleaned_texts)
        else:
            preds = model.predict(features)

        n_total = len(preds)
        n_real = int((preds == 0).sum())
        n_fake = int((preds == 1).sum())

        results[name] = {
            'sample_size': n_total,
            'predicted_real_count': n_real,
            'predicted_real_pct': round(n_real / n_total, 4) if n_total else 0.0,
            'predicted_fake_count': n_fake,
            'predicted_fake_pct': round(n_fake / n_total, 4) if n_total else 0.0,
        }
    return results


def run_full_external_evaluation(project_dir, external_csv_path, satire_csv_path=None, strip_cues=True):
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
    if 'title' in external_df.columns and 'text' in external_df.columns:
        external_df['text'] = external_df['title'].astype(str) + ' ' + external_df['text'].astype(str)

    in_domain_results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            in_domain_results = json.load(f)

    evaluation_report = {}
    comparison_table = []
    loaded_models = {}

    # 1. Classical models
    model_keys = list(MODEL_CONFIGS.keys()) + ['ensemble']
    for model_type in model_keys:
        filename = FakeNewsModel.get_filename(model_type)
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            continue

        display_name = FakeNewsModel.get_display_name(model_type)
        model = FakeNewsModel(model_type=model_type)
        model.load(model_path)
        loaded_models[display_name] = {'model': model, 'type': 'classical'}

        res = evaluate_on_external_dataset(
            display_name, model, 'classical', extractor, external_df, preprocessor,
        )
        evaluation_report[display_name] = res

        ext_acc = res['metrics']['accuracy']
        ci = res['metrics']['confidence_intervals']['accuracy_wilson_95ci']
        ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]"

        ext_macro_f1 = res['metrics']['macro_f1']
        in_acc = in_domain_results.get(display_name, {}).get('accuracy', np.nan)
        in_macro_f1 = in_domain_results.get(display_name, {}).get('macro_f1', in_domain_results.get(display_name, {}).get('f1', np.nan))

        acc_drop = in_acc - ext_acc if not np.isnan(in_acc) else np.nan
        f1_drop = in_macro_f1 - ext_macro_f1 if not np.isnan(in_macro_f1) else np.nan

        comparison_table.append({
            'Model': display_name,
            'In-Domain Acc': in_acc,
            'External Acc': ext_acc,
            'External Acc (95% CI)': ci_str,
            'Accuracy Drop (Δ)': acc_drop,
            'In-Domain Macro F1': in_macro_f1,
            'External Macro F1': ext_macro_f1,
            'Macro F1 Drop (Δ)': f1_drop,
        })

    # 2. Deep learning models if available
    lstm_path = os.path.join(models_dir, 'lstm_model.pt')
    lstm_tok = os.path.join(models_dir, 'lstm_tokenizer.joblib')
    if os.path.exists(lstm_path) and os.path.exists(lstm_tok):
        try:
            from deep_learning import LSTMModel
            lstm_model = LSTMModel()
            lstm_model.load(lstm_path, lstm_tok)
            display_name = 'Bidirectional LSTM'
            loaded_models[display_name] = {'model': lstm_model, 'type': 'deep_learning'}

            res = evaluate_on_external_dataset(
                display_name, lstm_model, 'deep_learning', extractor, external_df, preprocessor,
            )
            evaluation_report[display_name] = res

            ext_acc = res['metrics']['accuracy']
            ci = res['metrics']['confidence_intervals']['accuracy_wilson_95ci']
            ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            ext_macro_f1 = res['metrics']['macro_f1']
            in_acc = in_domain_results.get(display_name, {}).get('accuracy', np.nan)
            in_macro_f1 = in_domain_results.get(display_name, {}).get('macro_f1', in_domain_results.get(display_name, {}).get('f1', np.nan))
            acc_drop = in_acc - ext_acc if not np.isnan(in_acc) else np.nan
            f1_drop = in_macro_f1 - ext_macro_f1 if not np.isnan(in_macro_f1) else np.nan

            comparison_table.append({
                'Model': display_name,
                'In-Domain Acc': in_acc,
                'External Acc': ext_acc,
                'External Acc (95% CI)': ci_str,
                'Accuracy Drop (Δ)': acc_drop,
                'In-Domain Macro F1': in_macro_f1,
                'External Macro F1': ext_macro_f1,
                'Macro F1 Drop (Δ)': f1_drop,
            })
        except Exception as e:
            print(f"Warning: could not evaluate LSTM on external benchmark: {e}")

    trans_path = os.path.join(models_dir, 'transformer_model.pt')
    trans_tok = os.path.join(models_dir, 'transformer_tokenizer.joblib')
    if os.path.exists(trans_path) and os.path.exists(trans_tok):
        try:
            from deep_learning import TransformerModel
            trans_model = TransformerModel()
            trans_model.load(trans_path, trans_tok)
            display_name = 'Custom Transformer Encoder'
            loaded_models[display_name] = {'model': trans_model, 'type': 'deep_learning'}

            res = evaluate_on_external_dataset(
                display_name, trans_model, 'deep_learning', extractor, external_df, preprocessor,
            )
            evaluation_report[display_name] = res

            ext_acc = res['metrics']['accuracy']
            ci = res['metrics']['confidence_intervals']['accuracy_wilson_95ci']
            ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            ext_macro_f1 = res['metrics']['macro_f1']
            in_acc = in_domain_results.get(display_name, {}).get('accuracy', in_domain_results.get('Self-Attention Transformer', {}).get('accuracy', np.nan))
            in_macro_f1 = in_domain_results.get(display_name, {}).get('macro_f1', in_domain_results.get('Self-Attention Transformer', {}).get('macro_f1', np.nan))
            acc_drop = in_acc - ext_acc if not np.isnan(in_acc) else np.nan
            f1_drop = in_macro_f1 - ext_macro_f1 if not np.isnan(in_macro_f1) else np.nan

            comparison_table.append({
                'Model': display_name,
                'In-Domain Acc': in_acc,
                'External Acc': ext_acc,
                'External Acc (95% CI)': ci_str,
                'Accuracy Drop (Δ)': acc_drop,
                'In-Domain Macro F1': in_macro_f1,
                'External Macro F1': ext_macro_f1,
                'Macro F1 Drop (Δ)': f1_drop,
            })
        except Exception as e:
            print(f"Warning: could not evaluate Transformer on external benchmark: {e}")

    summary_df = pd.DataFrame(comparison_table)

    # 3. Satire Diagnostic Evaluation (if present)
    satire_report = {}
    if satire_csv_path and os.path.exists(satire_csv_path):
        satire_df = pd.read_csv(satire_csv_path)
        if 'title' in satire_df.columns and 'text' in satire_df.columns:
            satire_df['text'] = satire_df['title'].astype(str) + ' ' + satire_df['text'].astype(str)
        satire_report = evaluate_satire_benchmark(
            loaded_models, satire_df, preprocessor, extractor,
        )
        evaluation_report['satire_diagnostic_stress_test'] = satire_report

    return evaluation_report, summary_df, satire_report


def plot_generalization_drop(summary_df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if summary_df.empty:
        return

    plt.figure(figsize=(11, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    in_acc = summary_df['In-Domain Acc'].fillna(0)
    ext_acc = summary_df['External Acc'].fillna(0)

    plt.bar(x - width/2, in_acc, width, label='In-Domain Test (ISOT)', color='#4a9eff')
    plt.bar(x + width/2, ext_acc, width, label='External Benchmark (Unseen)', color='#e8a838')

    plt.xticks(x, summary_df['Model'], rotation=25, ha='right')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.15)
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
    parser.add_argument('--satire-benchmark', type=str, default='data/satire_benchmark.csv', help='Path to satire benchmark CSV')
    parser.add_argument('--keep-cues', action='store_true', help='Do not strip publisher cues during evaluation')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    benchmark_path = os.path.join(project_dir, args.benchmark) if not os.path.isabs(args.benchmark) else args.benchmark
    satire_path = os.path.join(project_dir, args.satire_benchmark) if not os.path.isabs(args.satire_benchmark) else args.satire_benchmark

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"External benchmark not found at {benchmark_path}")

    print(f"\n{'=' * 75}")
    print(f"EVALUATING MODELS ON INDEPENDENT EXTERNAL BENCHMARK")
    print(f"Benchmark: {benchmark_path}")
    print(f"{'=' * 75}\n")

    report, summary_df, satire_report = run_full_external_evaluation(
        project_dir, benchmark_path, satire_csv_path=satire_path, strip_cues=not args.keep_cues
    )

    print("--- Generalization Gap Summary (with Wilson 95% Confidence Intervals) ---")
    print(summary_df[['Model', 'In-Domain Acc', 'External Acc', 'External Acc (95% CI)', 'Accuracy Drop (Δ)', 'External Macro F1']].to_string(index=False))

    # Print detailed per-publisher breakdown for the primary baseline
    first_model = 'Logistic Regression' if 'Logistic Regression' in report else list(report.keys())[0]
    print(f"\n--- Per-Publisher Performance Breakdown ({first_model}) ---")
    src_data = report[first_model]['source_breakdown']
    breakdown_rows = []
    for pub, d in src_data.items():
        breakdown_rows.append({
            'Publisher': pub,
            'N': d['sample_size'],
            'True Real': d['real_count'],
            'True Fake': d['fake_count'],
            'Accuracy': f"{d['accuracy']:.1%}",
            '95% CI': f"[{d['wilson_95ci'][0]:.1%}, {d['wilson_95ci'][1]:.1%}]",
            'Macro F1': d['macro_f1'],
        })
    print(pd.DataFrame(breakdown_rows).to_string(index=False))

    if satire_report:
        print(f"\n--- Satire Diagnostic Stress Test (Out-of-Scope Diagnostic) ---")
        sat_rows = []
        for model_name, s_data in satire_report.items():
            sat_rows.append({
                'Model': model_name,
                'N': s_data['sample_size'],
                'Pred Real': f"{s_data['predicted_real_count']} ({s_data['predicted_real_pct']:.1%})",
                'Pred Fake': f"{s_data['predicted_fake_count']} ({s_data['predicted_fake_pct']:.1%})",
            })
        print(pd.DataFrame(sat_rows).to_string(index=False))
        print("\n*Note: Satire is isolated as an out-of-scope diagnostic stress test. Because satire adopts formal")
        print(" journalistic tone and grammatical polish, stylistic classifiers mistake it for real news.")

    out_json = os.path.join(models_dir, 'external_eval_results.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    plot_path = os.path.join(models_dir, 'external_generalization.png')
    plot_generalization_drop(summary_df, plot_path)

    print(f"\nReport saved to: {out_json}")
    print(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
