import os
from audio_analyser import audioAnalyser

BASE_PATH = "data/"
AUDIO_FILENAME = "audio2.wav"

if __name__ == "__main__":
    analysisJson = audioAnalyser.main(
        BASE_PATH=BASE_PATH,
        AUDIO_FILENAME=AUDIO_FILENAME,
        MODEL_SIZE="small",
        TOP_K=5,
        LONG_SENTENCE_THRESHOLD=25,
        TOP_PHRASES=5,
        return_json=True
    )

    print("Analysis completed successfully.")
    print(analysisJson)