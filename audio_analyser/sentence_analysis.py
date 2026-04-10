"""
Sentence structure and length analysis.
"""

import re
from .config import SENTENCE_PATTERN, WORD_PATTERN
from .text_processing import clean_text

def analyze_raw_sentences(transcript):
    """
    Detect natural speech issues without artificial splitting.
    """
    import re
    from .config import WORD_PATTERN

    # naive split (still useful for detection)
    rough_sentences = re.split(r'[.!?]+', transcript)
    rough_sentences = [s.strip() for s in rough_sentences if s.strip()]

    lengths = [len(WORD_PATTERN.findall(s)) for s in rough_sentences]

    if not lengths:
        return {
            "max_sentence_length": 0,
            "very_long_sentences": 0
        }

    return {
        "max_sentence_length": max(lengths),
        "very_long_sentences": sum(1 for l in lengths if l > 40)
    }

def split_into_sentences_by_length(words, max_len=20):
    """
    Split words into pseudo-sentences based on length.
    Acts as fallback when punctuation is unreliable.
    """
    sentences = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_len:
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences

def analyze_sentences(words, long_sentence_threshold=25):

    # Use chunk-based splitting instead of punctuation
    sentences = split_into_sentences_by_length(words, max_len=20)

    sentence_lengths = [len(s) for s in sentences]

    num_sentences = len(sentence_lengths)
    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0

    longest_length = max(sentence_lengths) if sentence_lengths else 0

    # Reconstruct longest sentence for preview
    longest_sentence = ""
    if sentences:
        longest_sentence = " ".join(max(sentences, key=len))

    long_sentence_count = sum(1 for l in sentence_lengths if l > long_sentence_threshold)
    long_sentence_percentage = (long_sentence_count / num_sentences * 100) if num_sentences > 0 else 0

    return (
        num_sentences,
        avg_sentence_length,
        longest_sentence,
        longest_length,
        long_sentence_count,
        long_sentence_percentage
    )