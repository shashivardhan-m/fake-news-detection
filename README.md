# Fake News Detection using NLP and Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

## 📋 Project Overview

An advanced AI-based Fake News Detection system that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as **Real** or **Fake**. This project demonstrates end-to-end ML workflow including data preprocessing, feature extraction, model training, evaluation, and deployment with a user-friendly web interface.

## ✨ Features

- **Multiple Machine Learning Models**: Logistic Regression and Random Forest
- **NLP Preprocessing**: Text cleaning, stopword removal, lemmatization
- **TF-IDF Feature Extraction**: Convert text to numerical vectors
- **Model Comparison**: Side-by-side performance evaluation
- **Interactive Web App**: Streamlit-based user interface
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualizations**: Comprehensive performance charts

## 📊 Dataset

This project uses the **ISOT Fake News Dataset** which contains:
- 21,417 real news articles from Reuters.com
- 23,481 fake news articles from various unreliable sources
- Total: 44,898 news articles

### Dataset Columns:
- `title`: News article title
- `text`: Full article text
- `subject`: News category
- `date`: Publication date
- `label`: 0 (Real) or 1 (Fake)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Programming Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy |
| **NLP** | NLTK |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Interface** | Streamlit |
| **Model Serialization** | Joblib |

## 📁 Project Structure

```
fake-news-detection/
├── app/                      # Web application
│   └── app.py               # Streamlit frontend
├── data/                     # Dataset directory (not tracked in git)
├── models/                   # Trained models (not tracked in git)
├── notebooks/                # Jupyter notebooks
├── src/                      # Source code
│   ├── preprocessing.py      # Text preprocessing
│   ├── feature_extraction.py # TF-IDF feature extraction
│   ├── models.py             # ML models
│   ├── deep_learning.py      # Deep learning models (LSTM)
│   ├── evaluation.py         # Model evaluation
│   ├── train.py              # Training pipeline
│   └── predict.py            # Prediction script
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. Download the ISOT Fake News Dataset
2. Place `True.csv` and `Fake.csv` in the `data/` directory

### Training the Models

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train both Logistic Regression and Random Forest models
- Evaluate model performance
- Save trained models to the `models/` directory

### Running the Web Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

### Command-Line Prediction

```bash
python src/predict.py "Your news article text here"
```

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 98.82%   | 99.31%    | 98.41% | 98.86%   |
| Random Forest       | 99.68%   | 99.68%    | 99.70% | 99.69%   |

## 🎯 Usage Examples

### Web Interface

1. Go to the **Predict** tab
2. Select a model (Logistic Regression or Random Forest)
3. Paste your news article text
4. Click "Check News"
5. View the prediction and confidence scores

### Model Comparison

Visit the **Model Comparison** tab to see:
- Detailed performance metrics table
- Interactive bar charts comparing all models
- Visualizations of accuracy, precision, recall, and F1-score

## 🔮 Future Improvements

- [ ] Implement BERT and other transformer models
- [ ] Add explainable AI (SHAP/LIME) for predictions
- [ ] Support for multiple languages
- [ ] Real-time news scraping
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] API endpoint for integration with other systems
- [ ] Batch processing for multiple articles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ISOT Fake News Dataset
- Scikit-learn documentation
- Streamlit community
- NLTK project

## 📞 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Made with ❤️ for AI/ML portfolio projects**

