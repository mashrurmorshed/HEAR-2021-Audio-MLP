import torch
from torch import nn
from torch.nn import functional as F
from audiomlp.models.audio_mae import gMLP_Encoder
from nnAudio.Spectrogram import MelSpectrogram
from einops import rearrange


class AudioMAE_Wrapper(nn.Module):
    """Wrapper for AudioMAE Encoder."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        timestamp_embedding_size: int = 4,
        scene_embedding_size: int = 792,
        encoder_params: dict = {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "dim": 64,
            "embed_dim": 4,
            "embed_drop": 0.0,
            "depth": 6,
            "ff_mult": 4,
            "prob_survival": 0.9,
            "pre_norm": False
        }
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.timestamp_embedding_size = timestamp_embedding_size
        self.scene_embedding_size = scene_embedding_size
        self.encoder = gMLP_Encoder(**encoder_params)
        self.audio_processor = MelSpectrogram(
            sr=sample_rate,
            n_mels=40,
            n_fft=480,
            win_length=480,
            hop_length=160,
            center=False
        )

    def forward(self, x: torch.Tensor):
        b, num_samples = x.shape
        
        # model input must be a multiple of sr
        if num_samples < self.sample_rate:
            x = F.pad(x, (0, self.sample_rate - num_samples), "constant", 0)
        elif num_samples %  self.sample_rate:
            x = F.pad(x, (0, self.sample_rate - num_samples %  self.sample_rate), "constant", 0)
        
        x = rearrange(x, "b (t sr) -> (b t) sr", sr=self.sample_rate)
        x = self.audio_processor(x)
        x = self.encoder(x)

        # compensate for chunking
        x = rearrange(x, "b d f -> b f d") # replicate cannot pad arbitrary axis
        x = F.pad(x, (1, 1), mode="replicate")
        x = rearrange(x, "(b t) f d -> b (t d) f", b=b)
        x = x[:, 1:-1, :]            
        return x




