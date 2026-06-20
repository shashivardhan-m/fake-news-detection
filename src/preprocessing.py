import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for resource in ('stopwords', 'wordnet', 'omw-1.4'):
    try:
        if resource == 'stopwords':
            stopwords.words('english')
        else:
            WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am", "it's": "it is", "that's": "that is",
}


class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

    def _expand_contractions(self, text):
        for contraction, expansion in CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = self._expand_contractions(text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 1 and token not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def preprocess_dataframe(self, df, text_column):
        df = df.copy()
        df[text_column + '_clean'] = df[text_column].apply(self.preprocess)
        return df
