"""
Text cleaning and pattern matching utilities.
"""

from .config import WORD_PATTERN

def clean_text(text):
    """
    Remove punctuation and return list of lowercase words.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of lowercase words
    """
    return WORD_PATTERN.findall(text.lower())

def extract_words_with_case(text):
    """
    Extract words preserving original casing.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of words with original casing
    """
    return WORD_PATTERN.findall(text)