import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
import numpy as np
from typing import Tuple

from .models import AudioMAE_Wrapper


def load_model(model_file_path: str) -> nn.Module:
    """Loads model weights from provided path.

    Args:
        model_file_path (str): Provided checkpoint path.

    Returns:
        nn.Module: Model instance.
    """
    model = AudioMAE_Wrapper()
    
    assert isinstance(model, nn.Module)
    assert hasattr(model, "sample_rate")
    assert hasattr(model, "timestamp_embedding_size")
    assert hasattr(model, "scene_embedding_size")

    ckpt = torch.load(model_file_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model


def initial_padding(audio: Tensor, sr=16000, hop_ms=10, window_ms=30):
    """Do some initial padding in order to get embeddings at the start/end of audio.

    Args:
        audio (Tensor): n_sounds x n_samples of mono audio.
        sr (int, optional): Sample rate. Defaults to 16000.
        hop_ms (int, optional): Hop length in ms. Defaults to 10.
        window_ms (int, optional): Window length in ms. Defaults to 30.

    Returns:
        Tensor: n_sounds x n_samples_padded.
    """
    init_pad = int((window_ms // 2 - hop_ms) / 1000 * sr) if window_ms // 2 > hop_ms else 0
    end_pad = int((window_ms // 2 ) / 1000 * sr)
    return F.pad(audio, (init_pad, end_pad), "constant", 0)
    

@torch.no_grad()
def get_timestamp_embeddings(audio: Tensor, model: nn.Module) -> Tuple[Tensor, Tensor]:
    """Returns embeddings at regular intervals centered at timestamps, as well as the timestamps themselves.

    Args:
        audio (Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embeddings (Tensor): A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps (Tensor): A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """

    sr = model.sample_rate
    window_ms = 30
    hop_ms = 10

    audio = initial_padding(audio, sr=sr, hop_ms=hop_ms, window_ms=window_ms)

    t_ms = 1000 * audio.shape[1] / sr
    n = int((t_ms - window_ms) / hop_ms + 1)
    init = hop_ms

    timestamps = torch.linspace(init, init + (n - 1) * hop_ms, n).expand(audio.shape[0], -1).float()
    embeddings = model(audio)

    assert embeddings.shape[1] >= n, "Sanity check."

    # truncate additional timesteps caused by padding
    embeddings = embeddings[:, :n, :]

    return embeddings, timestamps


@torch.no_grad()
def get_scene_embeddings(audio: Tensor, model: nn.Module) -> Tensor:
    """Returns a single embedding for the entire audio clip.

    Args:
        audio (Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embedding (Tensor): A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    sr = model.sample_rate
    window_ms = 30
    hop_ms = 10

    audio = initial_padding(audio, sr=sr, hop_ms=hop_ms, window_ms=window_ms)
    
    embed_t = model.scene_embedding_size // model.timestamp_embedding_size # 198 (2s)
    embeddings = model(audio) # (b, t, f) 
    b, t, f = embeddings.shape

    if t < embed_t:   # pad to embed_t
        embeddings = F.pad(embeddings, (0, 0, 0, embed_t - t), "constant", 0)
    
    elif t > embed_t: # temporal interpolation
        embeddings = rearrange(embeddings, "b t f -> b f t")
        # embeddings = F.interpolate(embeddings, size=embed_t, mode="linear", align_corners=True)
        # Decided to do repeated downsampling instead! Refer to: https://twitter.com/rzhang88/status/1258222917986312195?lang=en

        power_of_two = np.log2(t / embed_t)
        downsamp_reps = int(np.floor(power_of_two))
        rem = power_of_two - downsamp_reps
        
        for i in range(downsamp_reps):                   
            embeddings = F.interpolate(embeddings, size=embeddings.shape[-1]//2, mode="linear", align_corners=True)
            
        if rem > 0:    
            embeddings = F.interpolate(embeddings, size=embed_t, mode="linear", align_corners=True)
        
        embeddings = rearrange(embeddings, "b f t -> b t f")
        
    # flatten but maintain temporal order    
    embeddings = rearrange(embeddings, "b t f -> b (f t)")
    return embeddings