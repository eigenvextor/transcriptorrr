import json
import subprocess
import config

def download_media(media_url):
    # single video download
    # path is fine; since app is run from root dir
    path = config.MEDIA_DIR
    v_path = path / "videos"
    path.mkdir(exist_ok=True, parents=True)
    v_path.mkdir(exist_ok=True, parents=True)
    entries = path / "entries.json"
    entries.touch()

    media_id = media_url.split('/')[-1].split('=')[-1]

    with open(entries) as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print("initializing the file")
            data = []

    alr_downloaded = False
    for d in data:
        if media_id in d["media_id"]:
            return "media already downloaded!"

    if not alr_downloaded:
        print("downloading media...")
        title_command = [
                'yt-dlp',
                '--js-runtimes', 'node',
                '--remote-components', 'ejs:github',
                "--print", "title",
                media_url
            ]
        
        result = subprocess.run(title_command, capture_output=True, text=True)
        media_title = result.stdout.strip()
        print("title of the media: ", media_title)
    
        command = [
            "yt-dlp",
            "--js-runtimes", "node",
            "-t", "mp4",
            "--remote-components", "ejs:github",
            "-P", f"{v_path}",
            "-o", f"{media_id}.%(ext)s",
            # "-q", # quiet mode
            media_url
        ]

        process = subprocess.run(command)

        if process.returncode == 0:
            temp = {}
            id = len(data)
            temp["id"] = id + 1
            temp["media_id"] = media_id
            temp["title"] = media_title
            temp["transcription_flag"] = 0
            data.append(temp)

            with open(entries, 'w') as f:
                json.dump(data, f)

            return "media downloaded successfully!"

        else:
            return "error: yt-dlp failed to download the media. not saving to JSON."

