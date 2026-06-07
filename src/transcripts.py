from model import DiarizationModel, TranscriptionModel

whisper_model_id = "openai/whisper-medium.en"
tmodel = TranscriptionModel(whisper_model_id)
dmodel = DiarizationModel()

media_id = str(input("Type media-id: "))
transcripts, timestamps = tmodel.transcribe(media_id)

result = dmodel.diarization(media_id, timestamps, 2)
print(result)