import wave
import contextlib
import subprocess
from pathlib import Path

import config

def get_wav_path(m_id):
    path = config.MEDIA_DIR
    audio_path = path / f"videos/{m_id}.wav"
    video_path = path / f"videos/{m_id}.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file {m_id}.mp4 doesn't exist")
        return
        
    if Path(audio_path).exists():
        print(f"Audio file {m_id}.wav already exists")
        return audio_path
    else:
        command = f"ffmpeg -i {video_path} {audio_path}"
        try:
            _ = subprocess.run(command, shell=True, capture_output=True)
            print(f"Audio file {m_id}.wav created")
            return audio_path
        except:
            print(f"Couldn't convert {m_id}.mp4 video to .wav format")
            return

def get_duration(path):
  with contextlib.closing(wave.open(str(path),'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)

def validate_json():
    pass