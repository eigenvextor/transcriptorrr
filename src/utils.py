import subprocess
from pathlib import Path

import wave
import contextlib

def get_wav_path(m_id):
    path = str(Path(__file__).parent.parent)
    audio_path = f"{path}/audios/{m_id}.wav"
    video_path = f"{path}/videos/{m_id}.mp4"
    
    if not Path(video_path).exists():
        print(f"video file {m_id}.mp4 doesnt exist")
        return
        
    if Path(audio_path).exists():
        print(f"audio file {m_id}.wav already exists")
        return audio_path
    else:
        command = f"ffmpeg -i {video_path} {audio_path}"
        try:
            _ = subprocess.run(command, shell=True, capture_output=True)
            print(f"audio file {m_id}.wav created")
            return audio_path
        except:
            print(f"couldn't convert {m_id}.mp4 video to .wav format")
            return

def get_duration(path):
  with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)