import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from . import utils

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
        timestamps = result["chunks"]
        return transcripts, timestamps
        

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

        self.model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )