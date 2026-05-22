import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import joblib

class LSTMModel:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.is_fitted = False
    
    def preprocess_texts(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return padded
    
    def fit(self, texts, y, validation_split=0.2, epochs=10, batch_size=32):
        self.tokenizer.fit_on_texts(texts)
        X = self.preprocess_texts(texts)
        
        self.model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        self.is_fitted = True
    
    def predict(self, texts):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X = self.preprocess_texts(texts)
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, texts):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X = self.preprocess_texts(texts)
        probabilities = self.model.predict(X)
        return np.hstack([1 - probabilities, probabilities])
    
    def save(self, model_path, tokenizer_path):
        self.model.save(model_path)
        joblib.dump(self.tokenizer, tokenizer_path)
    
    def load(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.is_fitted = True

