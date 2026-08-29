import datetime
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src import utils


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

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,
            model_kwargs={
                "low_cpu_mem_usage": True, 
                "use_safetensors": True, # needed along w low_cpu_mem_usage for performance
            }
        )
        

    def transcribe(self, m_id):
        path = utils.get_wav_path(m_id)
        result = self.pipe(str(path)) # expects str/ array
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
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            # run_opts={"device": self.device}
        )

    def diarization(self, m_id, timestamps, num_speakers, metric="euclidean", linkage="ward"):
        chunks = timestamps.copy()
        num_speakers = min(max(round(int(num_speakers)), 1), len(chunks))
        if len(chunks) == 1:
            chunks[0]["speaker"] = "SPEAKER 1"
        else:
            path = utils.get_wav_path(m_id)
            duration = utils.get_duration(path)
            embeddings = np.zeros(shape=(len(chunks), 192))
            print(f"duration: {duration}")
            false_ids = []
            for i, chunk in enumerate(chunks):
                start = chunk["timestamp"][0]
                # whisper sometimes overshoots end timestamp of last chunk
                end = min(duration, chunk["timestamp"][1])
                # print(i, start, end)
                if start <= end:
                    clip = Segment(start, end)
                    waveform, _ = self.audio.crop(path, clip)
                    # we need mono channel audio
                    # print(f"waveform shape: {waveform.shape}")
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(axis=0, keepdim=True)
                    # print(f"waveform shape: {waveform.shape}")
                    embeddings[i] = self.model.encode_batch(waveform) # batch_size, num_channels, num_samples = waveforms.shape req
                    # print(f"embeddings[i]: {embeddings[i].shape}, embeddings: {embeddings.shape}")
                else:
                    print(f"WARNING: {i}, {chunk}")
                    false_ids.append(i)
            print(embeddings.shape)
            embeddings = np.nan_to_num(embeddings)
            embeddings = np.delete(embeddings, false_ids, axis=0)
            chunks = [chunk for i, chunk in enumerate(chunks) if i not in false_ids]
            print(embeddings.shape)

            # add speaker labels
            clustering = AgglomerativeClustering(num_speakers, metric=metric, linkage=linkage).fit(embeddings)
            # clustering = DBSCAN().fit(embeddings)
            labels = clustering.labels_
            for i in range(len(chunks)):
                chunks[i]["speaker"] = f"SPEAKER {(labels[i]+1)}"

            # output = ""
            # for (i, chunk) in enumerate(chunks):
            #     if i==0 or chunks[i-1]["speaker"] != chunk["speaker"]:
            #         if i!= 0:
            #             output += "\n\n"
            #         output += chunk["speaker"] + " " + str(datetime.timedelta(seconds=round(chunk["timestamp"][0]))) + "\n\n"
            #     output += chunk["text"][1:] + " "

            return chunks