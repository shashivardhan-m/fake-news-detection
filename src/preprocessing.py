import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

for resource in ('stopwords', 'wordnet', 'omw-1.4'):
    try:
        if resource == 'stopwords':
            stopwords.words('english')
        else:
            WordNetLemmatizer().lemmatize('test')
    except (LookupError, OSError):
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        except Exception:
            pass

CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
}

NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'cannot', 'barely',
    'hardly', 'scarcely', 'without', 'none', 'nobody', 'nowhere', 'nothing',
}

# Regex patterns matching news agency datelines and publisher artifacts
REUTERS_DATELINE_PATTERNS = [
    # E.g., "WASHINGTON (Reuters) -" or "MOSCOW/GENEVA (Reuters) —"
    re.compile(r'^[A-Z\s,/–\-]+(?:\([A-Za-z\s,/–\-]+\))?\s*\(Reuters\)\s*[-—–]\s*', re.IGNORECASE),
    re.compile(r'^\(Reuters\)\s*[-—–]\s*', re.IGNORECASE),
    re.compile(r'\b(?:WASHINGTON|LONDON|PARIS|MOSCOW|BERLIN|GENEVA|BEIJING|NEW YORK)\s+\(Reuters\)\b', re.IGNORECASE),
    re.compile(r'\bReuters\b', re.IGNORECASE),
    # Editorial sign-offs
    re.compile(r'Reporting by\s+[^;.\n]+(?:;\s*Editing by\s+[^.\n]+)?\.?', re.IGNORECASE),
    re.compile(r'Editing by\s+[^.\n]+\.?', re.IGNORECASE),
]

FAKE_NEWS_ARTIFACT_PATTERNS = [
    re.compile(r'Featured image via\s+[^.\n]+', re.IGNORECASE),
    re.compile(r'Image via\s+[^.\n]+', re.IGNORECASE),
    re.compile(r'Photo by\s+[^.\n]+', re.IGNORECASE),
    re.compile(r'\(Tweet\)', re.IGNORECASE),
    re.compile(r'Read more at\s+[^.\n]+', re.IGNORECASE),
]


def strip_publisher_artifacts(text):
    """
    Remove publisher datelines, agency signatures, and source-specific artifacts.
    This neutralizes shortcuts (such as ISOT's '(Reuters) -' marker on real articles).
    """
    if not isinstance(text, str):
        return ""

    cleaned = text
    for pattern in REUTERS_DATELINE_PATTERNS:
        cleaned = pattern.sub(' ', cleaned)
    for pattern in FAKE_NEWS_ARTIFACT_PATTERNS:
        cleaned = pattern.sub(' ', cleaned)

    return re.sub(r'\s+', ' ', cleaned).strip()


class TextPreprocessor:
    """
    Text preprocessing pipeline with:
    - Configurable negation word preservation (e.g., 'not', 'no', 'never')
    - Optional number preservation (numbers carry vital factual/statistical context)
    - Optional publisher artifact stripping (to eliminate source leakage)
    - Lemmatization vs. Stemming vs. None comparison modes
    """
    def __init__(
        self,
        use_stemming=False,
        use_lemmatization=True,
        preserve_negation=True,
        preserve_numbers=True,
        strip_publisher_cues=False,
    ):
        try:
            base_stopwords = set(stopwords.words('english'))
        except Exception:
            base_stopwords = {
                'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'is', 'was',
                'and', 'or', 'as', 'by', 'with', 'from', 'this', 'that', 'it',
            }

        self.preserve_negation = preserve_negation
        if self.preserve_negation:
            self.stop_words = base_stopwords - NEGATION_WORDS
        else:
            self.stop_words = set(base_stopwords)

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.preserve_numbers = preserve_numbers
        self.strip_publisher_cues = strip_publisher_cues

    def _expand_contractions(self, text):
        for contraction, expansion in CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def preprocess(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.strip_publisher_cues:
            text = strip_publisher_artifacts(text)

        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = self._expand_contractions(text)
        text = re.sub(r'[^\w\s]', ' ', text)

        if not self.preserve_numbers:
            text = re.sub(r'\d+', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        tokens = [
            token for token in tokens
            if (len(token) > 1 or (self.preserve_numbers and token.isdigit()))
            and token not in self.stop_words
        ]

        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except Exception:
                pass

        return ' '.join(tokens)

    def preprocess_dataframe(self, df, text_column, output_column=None):
        df = df.copy()
        target_col = output_column if output_column else text_column + '_clean'
        df[target_col] = df[text_column].apply(self.preprocess)
        return df
