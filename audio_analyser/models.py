"""
Whisper model management with caching.
"""

import whisper
import torch

# Model cache
_MODEL_CACHE = {}

def get_device():
    """Determine if CUDA is available."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_or_load_model(model_size, device):
    """
    Load Whisper model once and cache it.
    
    Args:
        model_size (str): Model size (tiny, base, small, medium, large)
        device (str): Device to load model on (cuda/cpu)
    
    Returns:
        whisper.model.Whisper: Loaded model
    """
    cache_key = (model_size, device)
    if cache_key not in _MODEL_CACHE:
        print(f"Loading Whisper model ({model_size})...")
        _MODEL_CACHE[cache_key] = whisper.load_model(model_size).to(device)
    return _MODEL_CACHE[cache_key]