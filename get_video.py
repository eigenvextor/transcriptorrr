import json
from pathlib import Path
import yt_dlp

urls = []

path = Path('videos')
path.mkdir(exist_ok=True)
entries_file = path / 'entries.json'
entries_file.touch()

with open(entries_file) as f:
    try:
        data = json.load(f)
    except json.decoder.JSONDecodeError:
        print('your file is empty!')
        data = {}

for url in urls:

    ydl_opts={
        'quiet': True,
        'skip_download': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        id_ = str(info['id'])
        title = info['title']
        media_type = info['media_type']

        if media_type == 'video':
            media_path = str(path / f'v_{id_}.%(ext)s')
        if media_type == 'short':
            media_path = str(path / f's_{id_}.%(ext)s')

    if id_ not in data.keys():
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl':  media_path,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)
        
        data[id_] = title

    else:
        print('its already downloaded')

with open(entries_file, 'w') as f:
    json.dump(data, f)
