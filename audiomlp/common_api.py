import torch
from torch import nn
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
    ckpt = torch.load(model_file_path, map_location="cpu")
    model = AudioMAE_Wrapper()
    model.load_state_dict(ckpt["model_state_dict"])
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    return model


def get_timestamp_embeddings(audio: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns embeddings at regular intervals centered at timestamps, as well as the timestamps themselves.

    Args:
        audio (torch.Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embeddings (torch.Tensor): A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps (torch.Tensor): A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """

    t_ms = 1000 * audio.shape[1] / model.sample_rate
    window_ms = 30
    hop_ms = 10

    assert t_ms >= window_ms, f"audio must be at least {window_ms}ms, but got {t_ms}ms."

    n = int((t_ms - window_ms) / hop_ms + 1)
    init = 15

    timestamps = torch.linspace(init, init + (n - 1) * 10, n).expand(audio.shape[0], -1)
    embeddings = model(audio)

    assert embeddings.shape[1] >= n

    # truncate additional timesteps caused by padding
    embeddings = embeddings[:, :n, :]

    return embeddings, timestamps


def get_scene_embeddings(audio: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Returns a single embedding for the entire audio clip.

    Args:
        audio (torch.Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embedding (torch.Tensor): A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """

    embed_t = model.scene_embedding_size // model.timestamp_embedding_size # 198 -> 2s
    embeddings = model(audio) # (b, t, f) 
    b, t, f = embeddings.shape

    if t < embed_t: # pad to embed_t
        embeddings = F.pad(embeddings, (0, 0, 0, embed_t - t), "constant", 0)
    
    elif t > embed_t:
        embeddings = rearrange(embeddings, "b t f -> b f t")
        # embeddings = F.interpolate(embeddings, size=embed_t, mode="linear", align_corners=True)
        # Decided to do repeated downsampling instead. Refer to: https://twitter.com/rzhang88/status/1258222917986312195?lang=en

        power_of_two = np.log2(t / embed_t)
        downsamp_reps = int(np.floor(power_of_two))
        rem = power_of_two - downsamp_reps
        
        for i in range(downsamp_reps):                   
            embeddings = F.interpolate(embeddings, size=embeddings.shape[-1]//2, mode="linear", align_corners=True)
            
        if rem > 0:    
            embeddings = F.interpolate(embeddings, size=embed_t, mode="linear", align_corners=True)
        
        embeddings = rearrange(embeddings, "b f t -> b t f")
        
    embeddings = rearrange(embeddings, "b t f -> b (f t)")
    return embeddings