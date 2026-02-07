import os
import re
from collections import Counter
import whisper
import librosa
import torch
import convertor

# ----------------------------
# CONFIG
# ----------------------------
BASE_PATH = "data/"
AUDIO_PATH = os.path.join(BASE_PATH, "audio2.wav")
MODEL_SIZE = "small"

TOP_K = 5

# ----------------------------
# ENSURE AUDIO EXISTS
# ----------------------------
if not os.path.exists(AUDIO_PATH):
    print("Audio not found, converting from video...")
    AUDIO_PATH = convertor.convert(AUDIO_PATH)

# ----------------------------
# LOAD AUDIO (for duration only)
# ----------------------------
y, sr = librosa.load(AUDIO_PATH, sr=None)
audio_duration_sec = librosa.get_duration(y=y, sr=sr)

# ----------------------------
# DEVICE SETUP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"

print(f"Using device : {device}")
if device == "cuda":
    print(f"GPU detected : {torch.cuda.get_device_name(0)}")

# ----------------------------
# LOAD MODEL (once)
# ----------------------------
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE).to(device)

# ----------------------------
# TRANSCRIBE (FAST SETTINGS)
# ----------------------------
print("Transcribing audio...")
print(f"Audio path: {AUDIO_PATH}")

# Load audio as numpy array first
audio_array, _ = librosa.load(AUDIO_PATH, sr=16000)

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

transcript = result["text"].strip()

# ----------------------------
# WORD COUNT (CLEAN)
# ----------------------------
# remove punctuation
clean_words = re.findall(r"\b[a-zA-Z']+\b", transcript.lower())

word_counts = Counter(clean_words)
top_words = word_counts.most_common(TOP_K)

# ----------------------------
# FLUENCY METRICS (approx)
# ----------------------------
total_words = len(clean_words)
wpm = (total_words / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0

# ----------------------------
# OUTPUT
# ----------------------------
print("\n==============================")
print("TRANSCRIPT")
print("==============================")
print(transcript)

print("\n==============================")
print("TOP USED WORDS")
print("==============================")
for word, count in top_words:
    print(f"{word:>10} : {count}")

print("\n==============================")
print("FLUENCY METRICS")
print("==============================")
print(f"Audio duration      : {audio_duration_sec:.2f} sec")
print(f"Total words         : {total_words}")
print(f"Words per minute    : {wpm:.2f}")