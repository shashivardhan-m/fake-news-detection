import argparse
import datetime
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from evaluation import ModelEvaluator
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from preprocessing import TextPreprocessor


def load_dataset(project_dir, sample_size=None, random_state=42):
    """
    Loads dataset: checks for True.csv & Fake.csv in root,
    falling back to data/sample_isot.csv if root files are not yet present.
    """
    true_path = os.path.join(project_dir, 'True copy.csv')
    if not os.path.exists(true_path):
        true_path = os.path.join(project_dir, 'True.csv')
    fake_path = os.path.join(project_dir, 'Fake.csv')

    if os.path.exists(true_path) and os.path.exists(fake_path):
        print(f"Loading dataset from: {true_path} and {fake_path}")
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df['label'] = 0
        fake_df['label'] = 1
        df = pd.concat([true_df, fake_df], ignore_index=True)
    elif os.path.exists(os.path.join(project_dir, 'data', 'sample_isot.csv')):
        sample_path = os.path.join(project_dir, 'data', 'sample_isot.csv')
        print(f"Using representative sample dataset from: {sample_path}")
        df = pd.read_csv(sample_path)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Please place True.csv and Fake.csv in {project_dir}"
        )

    df = df.dropna(subset=['title', 'text']) if 'title' in df.columns and 'text' in df.columns else df.dropna()
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    print(f"Loaded {len(df)} articles (Real: {(df['label'] == 0).sum()}, Fake: {(df['label'] == 1).sum()})")
    return df


def train_sklearn_models(
    X_train_clean,
    X_test_clean,
    y_train,
    y_test,
    project_dir,
    run_cv=False,
    cv_folds=5,
    cv_repeats=1,
    random_state=42,
):
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    extractor = FeatureExtractor(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_features = extractor.fit_transform(X_train_clean)
    X_test_features = extractor.transform(X_test_clean)
    extractor.save(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))

    results = {}
    trained_models = {}

    for model_type in MODEL_CONFIGS:
        display_name = FakeNewsModel.get_display_name(model_type)
        is_baseline = (model_type == 'logistic_regression')
        header_title = f"{display_name} (Primary Baseline)" if is_baseline else display_name

        print(f"\n{'=' * 55}\nTraining {header_title}...\n{'=' * 55}")

        model = FakeNewsModel(model_type=model_type)
        model.train(X_train_features, y_train)
        model.save(os.path.join(models_dir, FakeNewsModel.get_filename(model_type)))

        y_pred = model.predict(X_test_features)
        y_proba = model.predict_proba(X_test_features)[:, 1]

        evaluator = ModelEvaluator(y_test, y_pred, y_proba)
        metrics = evaluator.get_metrics()

        # Run cross-validation if requested
        if run_cv:
            print(f"Running {cv_folds}-fold CV...")
            # For LinearSVC / Calibrated, evaluate inner model or calibrated wrapper
            cv_results = ModelEvaluator.cross_validate_metrics(
                model.model, X_train_features, y_train.values,
                cv=cv_folds, n_repeats=cv_repeats, random_state=random_state
            )
            metrics.update(cv_results)

        results[display_name] = metrics

        print(f"\n{display_name} Test Metrics:")
        for k in ['accuracy', 'macro_f1', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'brier_score']:
            if k in metrics:
                print(f"  {k:15s}: {metrics[k]:.4f}")

        # Save confusion matrix plot
        cm_path = os.path.join(models_dir, f"cm_{model_type}.png")
        evaluator.plot_confusion_matrix(save_path=cm_path, title=f"Confusion Matrix: {display_name}")

        trained_models[model_type] = model

    # Ensemble model
    print(f"\n{'=' * 55}\nTraining Ensemble (Soft-Voting)...\n{'=' * 55}")
    ensemble = FakeNewsModel(model_type='ensemble')
    ensemble.build_ensemble([
        ('lr', trained_models['logistic_regression']),
        ('rf', trained_models['random_forest']),
        ('gb', trained_models['gradient_boosting']),
    ])
    ensemble.train(X_train_features, y_train)
    ensemble.save(os.path.join(models_dir, 'ensemble.joblib'))

    y_pred = ensemble.predict(X_test_features)
    y_proba = ensemble.predict_proba(X_test_features)[:, 1]
    evaluator = ModelEvaluator(y_test, y_pred, y_proba)
    ens_metrics = evaluator.get_metrics()
    results['Ensemble (Voting)'] = ens_metrics

    print("\nEnsemble Metrics:")
    for k in ['accuracy', 'macro_f1', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'brier_score']:
        if k in ens_metrics:
            print(f"  {k:15s}: {ens_metrics[k]:.4f}")

    cm_path = os.path.join(models_dir, "cm_ensemble.png")
    evaluator.plot_confusion_matrix(save_path=cm_path, title="Confusion Matrix: Ensemble (Voting)")

    return results, extractor, trained_models


def train_lstm_model(X_train_raw, X_test_raw, y_train, y_test, project_dir, epochs=5):
    from deep_learning import LSTMModel

    models_dir = os.path.join(project_dir, 'models')
    print(f"\n{'=' * 55}\nTraining Bidirectional LSTM (Word Embeddings)...\n{'=' * 55}")

    lstm = LSTMModel(max_words=10000, max_len=200, embedding_dim=128, hidden_dim=64)
    lstm.fit(X_train_raw.tolist(), y_train.values, epochs=epochs, batch_size=32)

    model_path = os.path.join(models_dir, 'lstm_model.pt')
    tokenizer_path = os.path.join(models_dir, 'lstm_tokenizer.joblib')
    lstm.save(model_path, tokenizer_path)

    y_pred = lstm.predict(X_test_raw.tolist())
    y_proba = lstm.predict_proba(X_test_raw.tolist())[:, 1]
    evaluator = ModelEvaluator(y_test, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    print("\nBidirectional LSTM Metrics:")
    for k in ['accuracy', 'macro_f1', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'brier_score']:
        if k in metrics:
            print(f"  {k:15s}: {metrics[k]:.4f}")

    cm_path = os.path.join(models_dir, "cm_lstm.png")
    evaluator.plot_confusion_matrix(save_path=cm_path, title="Confusion Matrix: Bidirectional LSTM")

    return {'Bidirectional LSTM': metrics}


def train_transformer_model(X_train_raw, X_test_raw, y_train, y_test, project_dir, epochs=5):
    from deep_learning import TransformerModel

    models_dir = os.path.join(project_dir, 'models')
    print(f"\n{'=' * 55}\nTraining Self-Attention Transformer Classifier...\n{'=' * 55}")

    trans = TransformerModel(max_words=10000, max_len=200, embedding_dim=128, nhead=4, num_layers=2)
    trans.fit(X_train_raw.tolist(), y_train.values, epochs=epochs, batch_size=32)

    model_path = os.path.join(models_dir, 'transformer_model.pt')
    tokenizer_path = os.path.join(models_dir, 'transformer_tokenizer.joblib')
    trans.save(model_path, tokenizer_path)

    y_pred = trans.predict(X_test_raw.tolist())
    y_proba = trans.predict_proba(X_test_raw.tolist())[:, 1]
    evaluator = ModelEvaluator(y_test, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    print("\nTransformer Metrics:")
    for k in ['accuracy', 'macro_f1', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'brier_score']:
        if k in metrics:
            print(f"  {k:15s}: {metrics[k]:.4f}")

    cm_path = os.path.join(models_dir, "cm_transformer.png")
    evaluator.plot_confusion_matrix(save_path=cm_path, title="Confusion Matrix: Transformer")

    return {'Self-Attention Transformer': metrics}


def save_comparison_plot(results, project_dir):
    results_df = pd.DataFrame(results).T
    metrics = ['accuracy', 'macro_f1', 'f1', 'pr_auc', 'roc_auc']
    available = [m for m in metrics if m in results_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, metric in enumerate(available):
        ax = axes[i]
        sns.barplot(
            x=results_df.index, y=results_df[metric], hue=results_df.index,
            ax=ax, palette='viridis', legend=False
        )
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ymin = max(0.5, results_df[metric].min() - 0.05) if not results_df[metric].isna().all() else 0.5
        ax.set_ylim([ymin, 1.02])
        ax.tick_params(axis='x', rotation=35)
        for patch in ax.patches:
            val = patch.get_height()
            if not np.isnan(val):
                ax.annotate(
                    f'{val:.3f}',
                    (patch.get_x() + patch.get_width() / 2, val),
                    ha='center', va='bottom', fontsize=8,
                )

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, 'models', 'comparison.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train fake news detection models with rigorous evaluation')
    parser.add_argument('--sample', type=int, default=None, help='Use a random subset for faster training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--strip-cues', action='store_true', default=True, help='Strip publisher datelines and source artifacts')
    parser.add_argument('--no-strip-cues', action='store_false', dest='strip_cues', help='Keep publisher cues (to replicate baseline bias)')
    parser.add_argument('--cv', action='store_true', default=True, help='Run k-fold cross validation')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--train-lstm', action='store_true', help='Train PyTorch BiLSTM model')
    parser.add_argument('--train-transformer', action='store_true', help='Train PyTorch Transformer model')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for deep learning models')
    parser.add_argument('--external-benchmark', type=str, default='data/external_benchmark.csv', help='Path to external benchmark CSV')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'=' * 65}")
    print("FAKE NEWS DETECTION RIGOROUS EXPERIMENT PIPELINE")
    print(f"Random Seed: {args.seed} | Strip Cues: {args.strip_cues} | CV: {args.cv} ({args.cv_folds} folds)")
    print(f"{'=' * 65}\n")

    # 1. Load dataset
    df = load_dataset(project_dir, sample_size=args.sample, random_state=args.seed)

    # 2. Preprocessing
    print(f"Preprocessing text (Preserve negation: True, Preserve numbers: True, Strip cues: {args.strip_cues})...")
    preprocessor = TextPreprocessor(
        preserve_negation=True,
        preserve_numbers=True,
        strip_publisher_cues=args.strip_cues,
    )
    df = preprocessor.preprocess_dataframe(df, 'text')

    # 3. Stratified Split
    print("Splitting data (stratified 80/20 train/test split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=args.seed, stratify=df['label']
    )
    X_train_clean = df.loc[X_train_raw.index, 'text_clean']
    X_test_clean = df.loc[X_test_raw.index, 'text_clean']

    # 4. Train classical models
    results, extractor, trained_models = train_sklearn_models(
        X_train_clean, X_test_clean, y_train, y_test, project_dir,
        run_cv=args.cv, cv_folds=args.cv_folds, random_state=args.seed,
    )

    # 5. Deep Learning Models
    if args.train_lstm:
        lstm_res = train_lstm_model(
            X_train_raw, X_test_raw, y_train, y_test, project_dir, epochs=args.epochs
        )
        results.update(lstm_res)

    if args.train_transformer:
        trans_res = train_transformer_model(
            X_train_raw, X_test_raw, y_train, y_test, project_dir, epochs=args.epochs
        )
        results.update(trans_res)

    # 6. Save results & plots
    results_file = os.path.join(models_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    save_comparison_plot(results, project_dir)

    # 7. Experiment Config for Reproducibility
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'random_seed': args.seed,
        'sample_size': args.sample or len(df),
        'strip_cues': args.strip_cues,
        'preserve_negation': True,
        'preserve_numbers': True,
        'cv_enabled': args.cv,
        'cv_folds': args.cv_folds,
        'models_evaluated': list(results.keys()),
        'vectorizer': {
            'max_features': 10000,
            'ngram_range': [1, 2],
            'sublinear_tf': True,
        }
    }
    with open(os.path.join(models_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 8. External Benchmark Evaluation (if present)
    ext_path = os.path.join(project_dir, args.external_benchmark) if not os.path.isabs(args.external_benchmark) else args.external_benchmark
    if os.path.exists(ext_path):
        print(f"\n{'=' * 65}\nEvaluating on External Benchmark: {ext_path}\n{'=' * 65}")
        from external_eval import run_full_external_evaluation, plot_generalization_drop
        ext_report, summary_df = run_full_external_evaluation(project_dir, ext_path, strip_cues=args.strip_cues)
        print("\nGeneralization Summary (In-Domain vs External):")
        print(summary_df.to_string(index=False))

        ext_json = os.path.join(models_dir, 'external_eval_results.json')
        with open(ext_json, 'w') as f:
            json.dump(ext_report, f, indent=2)
        plot_generalization_drop(summary_df, os.path.join(models_dir, 'external_generalization.png'))

    print(f"\n{'=' * 65}\nIn-Domain Model Comparison\n{'=' * 65}")
    display_cols = ['accuracy', 'macro_f1', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
    df_res = pd.DataFrame(results).T
    avail_cols = [c for c in display_cols if c in df_res.columns]
    print(df_res[avail_cols].to_string())
    print(f"\nTraining pipeline completed successfully! Artifacts stored in '{models_dir}'.")


if __name__ == '__main__':
    main()
