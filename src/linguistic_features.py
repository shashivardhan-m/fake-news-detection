import re
import numpy as np
import pandas as pd


def analyze_text(text):
    """
    Computes surface-level linguistic and stylistic metrics:
    word count, sentence length, punctuation frequency, capitalization,
    and sensational vocabulary usage.
    """
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
    """
    Returns heuristic stylistic observations.
    IMPORTANT: These are stylistic heuristics, NOT evidence of falsehood.
    Genuine news can feature urgent headlines, while sophisticated fake news
    often mimics neutral, formal editorial prose.
    """
    flags = []
    if stats['exclamation_ratio'] > 0.5:
        flags.append('Elevated exclamation usage (common in informal/sensationalist writing)')
    if stats['caps_ratio'] > 0.15:
        flags.append('High proportion of capitalized letters (stylistic emphasis)')
    if stats['sensational_score'] > 0.02:
        flags.append('Sensational vocabulary detected (clickbait/tabloid style)')
    if stats['avg_sentence_length'] < 8 and stats['word_count'] > 20:
        flags.append('Short average sentence length (characteristic of tabloid/snippet formats)')
    if stats['unique_word_ratio'] < 0.4 and stats['word_count'] > 50:
        flags.append('Low lexical diversity (repetitive vocabulary patterns)')
    return flags


def evaluate_heuristic_risk_flags(df, text_column='text', label_column='label'):
    """
    Empirically validates heuristic risk flags against labelled data.
    Measures:
      - Trigger rate in Real vs Fake articles
      - Precision (if flag is active, how often is article actually fake?)
      - False Positive Rate on real news
    """
    records = []
    for _, row in df.iterrows():
        text = str(row[text_column])
        stats = analyze_text(text)
        flags = get_risk_flags(stats)

        records.append({
            'label': int(row[label_column]),
            'has_exclamation_flag': stats['exclamation_ratio'] > 0.5,
            'has_caps_flag': stats['caps_ratio'] > 0.15,
            'has_sensational_flag': stats['sensational_score'] > 0.02,
            'has_short_sentence_flag': stats['avg_sentence_length'] < 8 and stats['word_count'] > 20,
            'has_low_diversity_flag': stats['unique_word_ratio'] < 0.4 and stats['word_count'] > 50,
            'total_flags': len(flags),
            'any_flag': len(flags) > 0,
        })

    flag_df = pd.DataFrame(records)
    results = {}

    flag_names = [
        'has_exclamation_flag',
        'has_caps_flag',
        'has_sensational_flag',
        'has_short_sentence_flag',
        'has_low_diversity_flag',
        'any_flag',
    ]

    real_mask = flag_df['label'] == 0
    fake_mask = flag_df['label'] == 1

    for flag in flag_names:
        real_trigger_rate = flag_df.loc[real_mask, flag].mean()
        fake_trigger_rate = flag_df.loc[fake_mask, flag].mean()

        pred_positive = flag_df[flag]
        true_positive = pred_positive & fake_mask
        precision = true_positive.sum() / max(pred_positive.sum(), 1)

        results[flag] = {
            'real_trigger_rate': round(float(real_trigger_rate), 4),
            'fake_trigger_rate': round(float(fake_trigger_rate), 4),
            'precision_as_fake_detector': round(float(precision), 4),
            'false_positive_rate_on_real': round(float(real_trigger_rate), 4),
        }

    return results
