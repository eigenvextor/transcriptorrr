import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import utils

class TranscriptionModel():
    def __init__(self, model_name):
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16    
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                return_timestamps=True
        )
        

    def transcribe(self, m_id):
        path = utils.get_wav_path(m_id)
        result = self.pipe(path)
        transcripts = result["text"]
        chunks = result["chunks"]
        return transcripts, chunks
        

class DiarizationModel:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16    
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
        self.audio = Audio()
        self.model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )

    def diarization(self, m_id, chunks, num_speakers):
        num_speakers = min(max(round(num_speakers), 1), len(chunks))
        if len(chunks) == 1:
            chunks[0]["speaker"] = "SPEAKER 1"
        else:
            path = utils.get_wav_path(m_id)
            duration = utils.get_duration(path)
            embeddings = np.zeros(shape=(len(chunks), 192))
            for i, chunk in enumerate(chunks):
                start = chunk["timestamp"][0]
                # whisper sometimes overshoots end timestamp of last chunk
                end = min(duration, chunk["timestamp"][1])
                clip = Segment(start, end)
                waveform, _ = self.audio.crop(path, clip)
                # we need mono channel audio
                # print(waveform.shape)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(axis=0, keepdim=True)
                # print(waveform.shape)
                embeddings[i] = self.model(waveform[None]) # batch_size, num_channels, num_samples = waveforms.shape req
                # print(embeddings[i].shape, embeddings.shape)
            embeddings = np.nan_to_num(embeddings)
        
            # add speaker labels
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            # clustering = DBSCAN().fit(embeddings)
            labels = clustering.labels_
            for i in range(len(chunks)):
                chunks[i]["speaker"] = f"SPEAKER {(labels[i]+1)}"

            output = ""
            for (i, chunk) in enumerate(chunks):
                if i==0 or chunks[i-1]["speaker"] != chunk["speaker"]:
                    if i!= 0:
                        output += "\n\n"
                    output += chunk["speaker"] + " " + str(datetime.timedelta(seconds=round(chunk["timestamp"][0]))) + "\n\n"
                output += chunk["text"][1:] + " "
            return output