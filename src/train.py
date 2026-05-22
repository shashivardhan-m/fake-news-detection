import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import FakeNewsModel
from evaluation import ModelEvaluator

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    true_path = os.path.join(project_dir, 'True copy.csv')
    if not os.path.exists(true_path):
        true_path = os.path.join(project_dir, 'True.csv')
    
    fake_path = os.path.join(project_dir, 'Fake.csv')
    
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print(f"Error: Data files not found!")
        print(f"Looking for:")
        print(f"  - {os.path.join(project_dir, 'True.csv')} or {os.path.join(project_dir, 'True copy.csv')}")
        print(f"  - {fake_path}")
        print("Please ensure both files are in the project directory.")
        return
    
    print("Loading data...")
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    
    true_df['label'] = 0
    fake_df['label'] = 1
    
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.dropna()
    df['text'] = df['title'] + ' ' + df['text']
    
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, 'text')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42
    )
    
    results = {}
    
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    extractor = FeatureExtractor()
    X_train_features = extractor.fit_transform(X_train)
    X_test_features = extractor.transform(X_test)
    extractor.save(os.path.join(project_dir, 'models', 'tfidf_vectorizer.joblib'))
    
    lr_model = FakeNewsModel(model_type='logistic_regression')
    lr_model.train(X_train_features, y_train)
    lr_model.save(os.path.join(project_dir, 'models', 'logistic_regression.joblib'))
    
    lr_pred = lr_model.predict(X_test_features)
    lr_evaluator = ModelEvaluator(y_test, lr_pred)
    results['Logistic Regression'] = lr_evaluator.get_metrics()
    print("\nLogistic Regression Metrics:")
    for key, value in results['Logistic Regression'].items():
        print(f"{key}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    rf_model = FakeNewsModel(model_type='random_forest')
    rf_model.train(X_train_features, y_train)
    rf_model.save(os.path.join(project_dir, 'models', 'random_forest.joblib'))
    
    rf_pred = rf_model.predict(X_test_features)
    rf_evaluator = ModelEvaluator(y_test, rf_pred)
    results['Random Forest'] = rf_evaluator.get_metrics()
    print("\nRandom Forest Metrics:")
    for key, value in results['Random Forest'].items():
        print(f"{key}: {value:.4f}")
    
    with open(os.path.join(project_dir, 'models', 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    comparison_df = pd.DataFrame(results).T
    print("\n", comparison_df)
    
    print("\n" + "="*50)
    print("Training complete! Models saved in 'models' directory.")

if __name__ == "__main__":
    main()

