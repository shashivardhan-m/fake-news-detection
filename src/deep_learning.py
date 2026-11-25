import collections
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleTokenizer:
    """
    Lightweight, fast vocabulary tokenizer with sequence padding and truncation.
    Independent of external C-extensions, fully compatible with Python 3.10-3.13.
    """
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.is_fitted = False

    def fit_on_texts(self, texts):
        counter = collections.Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.lower().split())

        most_common = counter.most_common(self.max_words - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fitted = True

    def texts_to_sequences(self, texts):
        sequences = []
        unk_idx = self.word2idx['<unk>']
        for text in texts:
            if not isinstance(text, str):
                sequences.append([])
                continue
            seq = [self.word2idx.get(w, unk_idx) for w in text.lower().split()]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        padded = np.zeros((len(sequences), self.max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            truncated = seq[:self.max_len]
            padded[i, :len(truncated)] = truncated
        return padded


# ── PyTorch Neural Architectures ──────────────────────────────────────────────

class _BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=64, num_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (h_n, _) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        cat = torch.cat([h_forward, h_backward], dim=1)
        hidden = self.relu(self.fc1(self.dropout(cat)))
        logits = self.fc2(self.dropout(hidden))
        return logits


class _TransformerNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, nhead=4, num_layers=2, max_len=200, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        embedded = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        encoded = self.transformer(embedded)
        # Global mean pooling over non-padding tokens
        mask = (x != 0).unsqueeze(-1)
        sum_encoded = (encoded * mask).sum(dim=1)
        mean_pooled = sum_encoded / mask.sum(dim=1).clamp(min=1)
        hidden = self.relu(self.fc1(self.dropout(mean_pooled)))
        logits = self.fc2(self.dropout(hidden))
        return logits


# ── Model Wrapper Class ───────────────────────────────────────────────────────

class BaseDeepLearningClassifier:
    def __init__(self, model_class, max_words=10000, max_len=200, embedding_dim=128, **kwargs):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model_class = model_class
        self.model_kwargs = kwargs
        self.tokenizer = SimpleTokenizer(max_words=max_words, max_len=max_len)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.is_fitted = False

    def _prepare_inputs(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = self.tokenizer.pad_sequences(sequences)
        return torch.tensor(padded, dtype=torch.long)

    def fit(self, texts, y, validation_split=0.2, epochs=5, batch_size=32, lr=1e-3, verbose=True):
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit_on_texts(texts)

        X_tensor = self._prepare_inputs(texts)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)

        n_samples = len(texts)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_loader = DataLoader(
            TensorDataset(X_tensor[train_idx], y_tensor[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )

        vocab_size = min(len(self.tokenizer.word2idx), self.max_words)
        self.model = self.model_class(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            max_len=self.max_len,
            **self.model_kwargs
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    val_x, val_y = X_tensor[val_idx].to(self.device), y_tensor[val_idx].to(self.device)
                    val_logits = self.model(val_x)
                    val_loss = criterion(val_logits, val_y).item()
                    val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

    def predict(self, texts):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        proba = self.predict_proba(texts)
        return proba.argmax(axis=1)

    def predict_proba(self, texts):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.eval()
        X_tensor = self._prepare_inputs(texts).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def save(self, model_path, tokenizer_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'model_class': self.model_class.__name__,
            'embedding_dim': self.embedding_dim,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'kwargs': self.model_kwargs,
        }, model_path)
        joblib.dump(self.tokenizer, tokenizer_path)

    def load(self, model_path, tokenizer_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.tokenizer = joblib.load(tokenizer_path)
        self.max_words = checkpoint['max_words']
        self.max_len = checkpoint['max_len']
        self.embedding_dim = checkpoint['embedding_dim']

        vocab_size = min(len(self.tokenizer.word2idx), self.max_words)
        self.model = self.model_class(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            max_len=self.max_len,
            **checkpoint.get('kwargs', {})
        ).to(self.device)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.is_fitted = True


class LSTMModel(BaseDeepLearningClassifier):
    """Bidirectional LSTM with learned word embeddings."""
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128, hidden_dim=64):
        super().__init__(
            model_class=_BiLSTMNet,
            max_words=max_words,
            max_len=max_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )


class TransformerModel(BaseDeepLearningClassifier):
    """Transformer with multi-head self-attention and positional encodings."""
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128, nhead=4, num_layers=2):
        super().__init__(
            model_class=_TransformerNet,
            max_words=max_words,
            max_len=max_len,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_layers=num_layers,
        )
