"""
Audio transcription logic.
"""

import os
import hashlib
import librosa
from .models import get_device, get_or_load_model

def transcribe_audio(audio_array, model_size="small"):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_array (np.ndarray): Audio data
        model_size (str): Whisper model size
    
    Returns:
        str: Transcribed text
    """
    device = get_device()
    use_fp16 = device == "cuda"

    print(f"Using device : {device}")
    if device == "cuda":
        import torch
        print(f"GPU detected : {torch.cuda.get_device_name(0)}")

    model = get_or_load_model(model_size, device)

    print("Transcribing audio...")
    
    result = model.transcribe(
        audio_array,
        language='en',
        verbose=False,
        fp16=use_fp16,
        word_timestamps=False,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False
    )

    return result["text"].strip()

def hash_file(filepath):
    """
    Generate SHA-256 hash of a file.
    
    Args:
        filepath (str): Path to file
    
    Returns:
        str: Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_transcript(transcript, base_path, audio_filename):
    """
    Save transcript to file and generate hash.
    
    Args:
        transcript (str): Transcript text
        base_path (str): Base directory path
        audio_filename (str): Original audio filename
    """
    # Save transcript
    transcript_path = os.path.join(
        base_path, 
        f"transcript_{audio_filename.replace('.wav', '')}.txt"
    )
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Generate and save hash
    file_hash = hash_file(transcript_path)
    hash_path = os.path.join(
        base_path,
        f"transcript_{audio_filename.replace('.wav', '')}.txt.sha256"
    )
    with open(hash_path, "w") as f:
        f.write(file_hash)
    
    print(f"  ✓ Transcript saved!")
    print(f"  🔒 Hash: {file_hash[:16]}...")

def load_transcript(base_path, audio_filename):
    """
    Load existing transcript from file.
    
    Args:
        base_path (str): Base directory path
        audio_filename (str): Original audio filename
    
    Returns:
        str or None: Transcript text if exists, None otherwise
    """
    transcript_path = os.path.join(
        base_path,
        f"transcript_{audio_filename.replace('.wav', '')}.txt"
    )
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def verify_transcript_integrity(base_path, audio_filename):
    """
    Verify transcript hasn't been tampered with using hash.
    
    Args:
        base_path (str): Base directory path
        audio_filename (str): Original audio filename
    
    Returns:
        bool: True if hash matches, False otherwise
    """
    transcript_path = os.path.join(
        base_path,
        f"transcript_{audio_filename.replace('.wav', '')}.txt"
    )
    hash_path = os.path.join(
        base_path,
        f"transcript_{audio_filename.replace('.wav', '')}.txt.sha256"
    )
    
    if not os.path.exists(transcript_path) or not os.path.exists(hash_path):
        return False
    
    # Read stored hash
    with open(hash_path, "r") as f:
        stored_hash = f.read().strip()
    
    # Calculate current hash
    current_hash = hash_file(transcript_path)
    
    return stored_hash == current_hash