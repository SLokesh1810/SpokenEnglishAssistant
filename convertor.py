import subprocess

def convert(AUDIO_FILE):
    FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    VIDEO_PATH = AUDIO_FILE[:-3] + ".wav"
    AUDIO_PATH = AUDIO_FILE

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", VIDEO_PATH,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        AUDIO_PATH
    ]

    subprocess.run(command, check=True)
    print("Audio extracted successfully")
    return VIDEO_PATH
