import os

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-2026-02-04-git-627da1111c-full_build\bin"

import whisper
import librosa
import numpy as np
import torch
import convertor

# ----------------------------
# CONFIG
# ----------------------------
BASE_PATH = "data/"
AUDIO_PATH = os.path.join(BASE_PATH, "audio1.wav")
MODEL_SIZE = "small"   # tiny | base | small | medium

# ----------------------------
# ENSURE AUDIO EXISTS
# ----------------------------
if not os.path.exists(AUDIO_PATH):
    print("Audio not found, converting from video...")
    AUDIO_PATH = convertor.convert()

# ----------------------------
# LOAD AUDIO (for duration)
# ----------------------------
y, sr = librosa.load(AUDIO_PATH, sr=None)
audio_duration_sec = librosa.get_duration(y=y, sr=sr)

# ----------------------------
# DEVICE SETUP (GPU / CPU)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"

print(f"Using device : {device}")
if device == "cuda":
    print(f"GPU detected : {torch.cuda.get_device_name(0)}")

# ----------------------------
# LOAD WHISPER MODEL
# ----------------------------
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE).to(device)

# ----------------------------
# TRANSCRIBE
# ----------------------------
print("Transcribing audio...")
result = model.transcribe(
    AUDIO_PATH,
    word_timestamps=True,
    verbose=False,
    fp16=use_fp16
)

transcript = result["text"].strip()

# ----------------------------
# WORD TIMESTAMPS
# ----------------------------
words = []
for segment in result.get("segments", []):
    for word in segment.get("words", []):
        words.append(word)

total_words = len(words)

# ----------------------------
# FLUENCY METRICS
# ----------------------------
# Words per minute
wpm = (total_words / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0

# Pause analysis
pauses = []
for i in range(1, len(words)):
    gap = words[i]["start"] - words[i - 1]["end"]
    if gap > 0:
        pauses.append(gap)

avg_pause = float(np.mean(pauses)) if pauses else 0.0
long_pauses = [p for p in pauses if p > 1.0]

# ----------------------------
# OUTPUT
# ----------------------------
print("\n==============================")
print("TRANSCRIPT")
print("==============================")
print(transcript)

print("\n==============================")
print("FLUENCY METRICS")
print("==============================")
print(f"Audio duration      : {audio_duration_sec:.2f} sec")
print(f"Total words         : {total_words}")
print(f"Words per minute    : {wpm:.2f}")
print(f"Average pause       : {avg_pause:.2f} sec")
print(f"Long pauses (>1s)   : {len(long_pauses)}")
