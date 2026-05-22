# Fake News Detection

An AI-based Fake News Detection system using Natural Language Processing (NLP) and Machine Learning.

## Project Structure

```
fake news detection/
├── data/                   # Directory for dataset files
├── models/                 # Directory for trained models and vectorizers
├── src/
│   ├── preprocessing.py    # Text preprocessing module
│   ├── feature_extraction.py  # TF-IDF feature extraction
│   ├── models.py           # ML models (Logistic Regression, Random Forest)
│   └── evaluation.py       # Model evaluation metrics
├── app.py                  # Streamlit frontend application
├── train.py                # Training pipeline
├── predict.py              # Command-line prediction script
└── requirements.txt        # Project dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Get the Kaggle Fake News Dataset from: https://www.kaggle.com/c/fake-news/data

Download `train.csv` and place it in the `data/` directory.

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Extract TF-IDF features
- Train a Logistic Regression model
- Evaluate the model
- Save the trained model and vectorizer in the `models/` directory

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

This will launch a web interface where you can enter news articles to check if they're real or fake.

### 5. Command-Line Prediction

```bash
python predict.py "Your news article text here"
```

## Models

- **Logistic Regression** (default)
- **Random Forest**

To use Random Forest, modify `train.py` and change `model_type='logistic_regression'` to `model_type='random_forest'`.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

