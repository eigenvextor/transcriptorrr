import json
import subprocess
from pathlib import Path

# single video download
path = Path('../media/videos')
path.mkdir(exist_ok=True, parents=True)
entries = path / 'entries.json'
entries.touch()

media_url = ""
media_id = media_url.split('/')[-1].split('=')[-1]

with open(entries) as f:
    try:
        data = json.load(f)
    except json.decoder.JSONDecodeError:
        print("Initializing the file")
        data = {}

title_command = [
    'yt-dlp',
    '--js-runtimes', 'node',
    '--remote-components', 'ejs:github',
    "--print", "title",
    media_url
]

result = subprocess.run(title_command, capture_output=True, text=True)
media_title = result.stdout.strip()

if media_id not in data.keys():
    command = [
        "yt-dlp",
        "--js-runtimes", "node",
        "-t", "mp4",
        "--remote-components", "ejs:github",
        "-P", f"{path}",
        "-o", f"{media_id}.%(ext)s",
        "-q", # quiet mode
        media_url
    ]

    process = subprocess.run(command)

    if process.returncode == 0:
        data[media_id] = media_title
        print("Media downloaded successfully!")

    else:
        print("Error: yt-dlp failed to download the media. Not saving to JSON.")

else:
    print("Media already downloaded!")


with open(entries, 'w') as f:
    json.dump(data, f)
