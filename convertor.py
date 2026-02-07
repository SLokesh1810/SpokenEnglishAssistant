import subprocess
import os

FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"

def convert(path: str) -> str:
    """
    Convert a video file (mp4/mkv/etc.) to mono 16kHz WAV.
    Returns path to generated WAV file.
    """

    base, _ = os.path.splitext(path)
    video_path = base + ".mp4"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    audio_path = base + ".wav"

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]

    subprocess.run(command, check=True)
    print(f"Audio extracted successfully → {audio_path}")

    return audio_path