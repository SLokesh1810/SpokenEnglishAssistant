"""
Word categorization and frequency analysis.
"""

from collections import Counter
from .config import WORD_CATEGORIES

def categorize_words(words):
    """
    Count words by category.
    
    Args:
        words (list): List of lowercase words
    
    Returns:
        dict: Category counts {category_name: count}
    """
    category_counts = {
        category: sum(1 for word in words if word in word_set)
        for category, word_set in WORD_CATEGORIES.items()
    }
    return category_counts

def get_top_words(words, top_k=5):
    """
    Get most frequent words.
    
    Args:
        words (list): List of words
        top_k (int): Number of top words to return
    
    Returns:
        list: List of (word, count) tuples
    """
    word_counts = Counter(words)
    return word_counts.most_common(top_k)