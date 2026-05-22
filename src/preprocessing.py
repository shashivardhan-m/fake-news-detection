import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir)

class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words]
        
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column):
        df[text_column + '_clean'] = df[text_column].apply(self.preprocess)
        return df

