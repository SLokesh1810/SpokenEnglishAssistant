"""
Entry point for audio analysis.
"""

import os
from audio_analyser import main

BASE_PATH = "data/"
AUDIO_FILENAME = "audio1.wav"

if __name__ == "__main__":
    main(
        base_path=BASE_PATH,
        audio_filename=AUDIO_FILENAME,
        model_size="small",
        top_k=3,
        long_sentence_threshold=25,
        top_phrases=3,
        return_json=False,
        save_json=True
    )