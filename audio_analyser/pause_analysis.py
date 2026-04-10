import librosa
import numpy as np

def analyze_pauses(audio_path, sr=16000, silence_threshold=20, min_silence_duration=0.5):
    """
    Detect pauses (silence) in audio.

    Args:
        audio_path (str): path to audio file
        sr (int): sampling rate
        silence_threshold (int): threshold in dB
        min_silence_duration (float): minimum silence (seconds)

    Returns:
        dict: pause metrics
    """

    y, sr = librosa.load(audio_path, sr=sr)

    # Convert threshold to amplitude
    intervals = librosa.effects.split(
        y,
        top_db=silence_threshold
    )

    total_duration = librosa.get_duration(y=y, sr=sr)

    speech_segments = []
    for start, end in intervals:
        speech_segments.append((start / sr, end / sr))

    # Calculate pauses between speech segments
    pauses = []
    for i in range(1, len(speech_segments)):
        prev_end = speech_segments[i-1][1]
        curr_start = speech_segments[i][0]

        pause_duration = curr_start - prev_end

        if pause_duration >= min_silence_duration:
            pauses.append(pause_duration)

    if not pauses:
        return {
            "total_pauses": 0,
            "long_pauses": 0,
            "avg_pause_duration": 0
        }

    return {
        "total_pauses": len(pauses),
        "long_pauses": sum(1 for p in pauses if p > 1.5),
        "avg_pause_duration": round(np.mean(pauses), 2)
    }