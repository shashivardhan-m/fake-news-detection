import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import FakeNewsModel, MODEL_CONFIGS
from evaluation import ModelEvaluator


def load_dataset(project_dir, sample_size=None):
    true_path = os.path.join(project_dir, 'True copy.csv')
    if not os.path.exists(true_path):
        true_path = os.path.join(project_dir, 'True.csv')

    fake_path = os.path.join(project_dir, 'Fake.csv')

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Dataset not found. Place True.csv and Fake.csv in {project_dir}"
        )

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df['label'] = 0
    fake_df['label'] = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.dropna()
    df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    return df


def train_sklearn_models(X_train, X_test, y_train, y_test, project_dir):
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    extractor = FeatureExtractor()
    X_train_features = extractor.fit_transform(X_train)
    X_test_features = extractor.transform(X_test)
    extractor.save(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))

    results = {}
    trained_models = {}

    for model_type in MODEL_CONFIGS:
        display_name = FakeNewsModel.get_display_name(model_type)
        print(f"\n{'=' * 50}\nTraining {display_name}...\n{'=' * 50}")

        model = FakeNewsModel(model_type=model_type)
        model.train(X_train_features, y_train)
        model.save(os.path.join(models_dir, FakeNewsModel.get_filename(model_type)))

        y_pred = model.predict(X_test_features)
        y_proba = model.predict_proba(X_test_features)[:, 1]
        evaluator = ModelEvaluator(y_test, y_pred, y_proba)
        results[display_name] = evaluator.get_metrics()

        print(f"\n{display_name} Metrics:")
        for key, value in results[display_name].items():
            print(f"  {key}: {value:.4f}")

        trained_models[model_type] = model

    print(f"\n{'=' * 50}\nTraining Ensemble (Voting)...\n{'=' * 50}")
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
    results['Ensemble (Voting)'] = evaluator.get_metrics()

    print("\nEnsemble Metrics:")
    for key, value in results['Ensemble (Voting)'].items():
        print(f"  {key}: {value:.4f}")

    return results, extractor


def train_lstm_model(X_train, X_test, y_train, y_test, project_dir, epochs=5):
    from deep_learning import LSTMModel

    models_dir = os.path.join(project_dir, 'models')
    print(f"\n{'=' * 50}\nTraining Bidirectional LSTM...\n{'=' * 50}")

    lstm = LSTMModel()
    lstm.fit(X_train.tolist(), y_train.values, epochs=epochs, batch_size=64)

    model_path = os.path.join(models_dir, 'lstm_model.keras')
    tokenizer_path = os.path.join(models_dir, 'lstm_tokenizer.joblib')
    lstm.save(model_path, tokenizer_path)

    y_pred = lstm.predict(X_test.tolist())
    y_proba = lstm.predict_proba(X_test.tolist())[:, 1]
    evaluator = ModelEvaluator(y_test, y_pred, y_proba)
    metrics = evaluator.get_metrics()

    print("\nLSTM Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return {'Bidirectional LSTM': metrics}


def save_comparison_plot(results, project_dir):
    results_df = pd.DataFrame(results).T
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available = [m for m in metrics if m in results_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, metric in enumerate(available):
        ax = axes[i]
        sns.barplot(x=results_df.index, y=results_df[metric], hue=results_df.index, ax=ax, palette='viridis', legend=False)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim([0.85, 1.0])
        ax.tick_params(axis='x', rotation=45)
        for patch in ax.patches:
            ax.annotate(
                f'{patch.get_height():.3f}',
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha='center', va='bottom', fontsize=8,
            )

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, 'models', 'comparison.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--sample', type=int, default=None, help='Use a random subset for faster training')
    parser.add_argument('--train-lstm', action='store_true', help='Also train the LSTM deep learning model')
    parser.add_argument('--lstm-epochs', type=int, default=5, help='Epochs for LSTM training')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    df = load_dataset(project_dir, sample_size=args.sample)

    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, 'text')

    print("Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label'],
    )

    results, _ = train_sklearn_models(X_train, X_test, y_train, y_test, project_dir)

    if args.train_lstm:
        lstm_results = train_lstm_model(
            X_train, X_test, y_train, y_test, project_dir, epochs=args.lstm_epochs,
        )
        results.update(lstm_results)

    with open(os.path.join(models_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    save_comparison_plot(results, project_dir)

    print(f"\n{'=' * 50}\nModel Comparison\n{'=' * 50}")
    print(pd.DataFrame(results).T.to_string())
    print(f"\nTraining complete! Models saved in '{models_dir}'")


if __name__ == "__main__":
    main()
