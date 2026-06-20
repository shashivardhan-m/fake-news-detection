import re


def analyze_text(text):
    if not isinstance(text, str) or not text.strip():
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'exclamation_ratio': 0.0,
            'question_ratio': 0.0,
            'caps_ratio': 0.0,
            'unique_word_ratio': 0.0,
            'sensational_score': 0.0,
        }

    words = re.findall(r'\b\w+\b', text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    sensational_words = {
        'shocking', 'unbelievable', 'breaking', 'exclusive', 'secret',
        'hoax', 'conspiracy', 'scandal', 'outrage', 'bombshell',
        'urgent', 'alert', 'exposed', 'destroy', 'lie', 'lies',
    }
    lower_words = [w.lower() for w in words]
    sensational_hits = sum(1 for w in lower_words if w in sensational_words)

    alpha_chars = [c for c in text if c.isalpha()]
    caps_chars = [c for c in alpha_chars if c.isupper()]

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0,
        'avg_sentence_length': round(word_count / sentence_count, 2),
        'exclamation_ratio': round(text.count('!') / sentence_count, 3),
        'question_ratio': round(text.count('?') / sentence_count, 3),
        'caps_ratio': round(len(caps_chars) / len(alpha_chars), 3) if alpha_chars else 0.0,
        'unique_word_ratio': round(len(set(lower_words)) / word_count, 3) if word_count else 0.0,
        'sensational_score': round(sensational_hits / word_count, 4) if word_count else 0.0,
    }


def get_risk_flags(stats):
    flags = []
    if stats['exclamation_ratio'] > 0.5:
        flags.append('High exclamation usage — common in clickbait')
    if stats['caps_ratio'] > 0.15:
        flags.append('Excessive capitalization detected')
    if stats['sensational_score'] > 0.02:
        flags.append('Sensational language patterns found')
    if stats['avg_sentence_length'] < 8:
        flags.append('Very short sentences — may indicate low-quality content')
    if stats['unique_word_ratio'] < 0.4 and stats['word_count'] > 50:
        flags.append('Low vocabulary diversity')
    return flags
