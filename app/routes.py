import re
import json
import time
import threading
from app import app
from flask import render_template, request, redirect, url_for, jsonify
from src.download import download_media
from src.model import DiarizationModel, TranscriptionModel
import config

# global variable to track
status = {
    "is_running": False,
    "message": ""
}

# load model

# not necessary as flask takes time anyway to spinup
# is_loading = True
model_id = "openai/whisper-medium.en"
tmodel = TranscriptionModel(model_id)
dmodel = DiarizationModel()

# load entry
entry_path = config.MEDIA_DIR / "entries.json"
if not entry_path.exists() or entry_path.stat().st_size == 0:
    with open(entry_path, "w") as f:
        json.dump([], f)

@app.route('/')
def index():
    # load the updated file everytime this entrypoint is hit 
    with open(entry_path, "r") as f:
        entries = json.load(f)
    # TODO make appropriate changes here
    return render_template("index.html", data=entries, title="transcriptorrr")

def process_media(media_url, media_id, no_speakers, is_downloaded):
    status["is_running"] = True

    try:
        # download if necessary
        if not is_downloaded:
            status["message"] = f"downloading media: {media_id}"
            download_message = download_media(media_url)
            status["message"] = download_message

        # transcribe and speaker diarization
        status["message"] = f"transcribing media: {media_id}"
        transcripts, timestamps = tmodel.transcribe(media_id)
        result = dmodel.diarization(media_id, timestamps, no_speakers)

        file_path = config.MEDIA_DIR / f"transcripts/{media_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)

        with open(entry_path, "r") as f:
            current_entries = json.load(f)

        for entry in current_entries:
            if entry.get("media_id") == media_id:
                entry["transcription_flag"] = 2
                break

        with open(entry_path, "w") as f:
            json.dump(current_entries, f, indent=4)

        status["messsage"] = f"everything done for media: {media_id}"

    except Exception as e:
        status["message"] = f"error processing {media_id}: {e}"

    finally:
        time.sleep(4) # js fetch status every 3s. 4s guarantees it
        status["is_running"] = False
        

@app.route('/generate-transcripts', methods=['GET', 'POST'])
def generate_transcripts():

    # load the updated file everytime this entrypoint is hit 
    with open(entry_path, "r") as f:
        entries = json.load(f)

    if request.method == "POST":
        is_media_downloaded, is_media_transcribed = False, False
        media_url = request.form.get("media_url")
        no_speakers = request.form.get("no_speakers")

        if not media_url:
            return redirect(url_for("generate_transcripts"))

        media_id = media_url.split('/')[-1].split('=')[-1]

        # check database if media is alr downloaded/ transcribed
        for entry in entries:
            if entry["media_id"] == media_id:
                is_media_downloaded = True
                if entry["transcription_flag"] == 2:
                    is_media_transcribed = True
                break # early stopping

        # check global lock
        if status["is_running"]:
            print("something is alr going on. please wait")
            return redirect(url_for("index"))

        # check if everything is done alr
        if is_media_transcribed:
            print("everything is alr done for current media")
            return redirect(url_for("handle_transcript", video_id=media_id))

        else:
            thread = threading.Thread(
                target=process_media,
                args=(media_url, media_id, no_speakers, is_media_downloaded,)
            )
            thread.start()

            return redirect(url_for("index"))

    # GET request
    return render_template("new.html",)


@app.route("/transcript/<video_id>", methods=["GET", "POST"])
def handle_transcript(video_id):
    file_path = config.MEDIA_DIR / f"transcripts/{video_id}.json"

    if request.method == "POST":
        raw_text = request.form.get("transcript_data")
        parsed_data = []

        # html use \r\n for linebreaks
        normalized_text = raw_text.replace("\r\n", "\n")
        blocks = re.split(r"\n{2,}", normalized_text.strip())

        try:
            for block in blocks:
                block = block.strip()
                if not block:
                    continue

                match = re.match(r"^\[(.*?)]:\s*(.*)$", block, re.DOTALL)

                if match:
                    parsed_data.append({
                        "speaker": match.group(1).strip(),
                        "text": match.group(2).strip()
                    })
                else:
                    raise ValueError(f"format error here: {block[:10]}")

            with open(file_path, "w") as f:
                json.dump(parsed_data, f, indent=4)

            return redirect(url_for("handle_transcript", video_id=video_id))

        except ValueError as e:
            return render_template(
                "transcript.html",
                data_str=raw_text,
                video_id=video_id,
                mode="edit",
                error=str(e)
            )
        
    # GET request
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 

    # view mode is the default
    mode = request.args.get("mode", "view")

    # json to custom text format if edit mode
    data_str = ""
    if mode == "edit":
        text_blocks = []
        for entry in data:
            speaker = entry.get("speaker", "UNKNOWN")
            text = entry.get("text", "")
            text_blocks.append(f"[{speaker}]: {text}")
        data_str = "\n\n".join(text_blocks)

    return render_template(
        "transcript.html",
        data = data,
        data_str = data_str,
        video_id = video_id,
        mode = mode
    )

@app.route("/api/status")
def get_status():
    return jsonify(status)