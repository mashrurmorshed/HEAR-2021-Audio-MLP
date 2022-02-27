"""*Common API for HEAR-2021@NeurIPS'21*"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
import numpy as np
from typing import Tuple

from kwmlp.models import AudioMLP_Wrapper
from kwmlp.utils import initial_padding


def load_model(model_file_path: str) -> nn.Module:
    """Loads model weights from provided path.

    Args:
        model_file_path (str): Provided checkpoint path.

    Returns:
        nn.Module: Model instance.
    """

    embed_dim = 64
    scene_dim = 1024
    encoder_type = "kwmlp"

    model = AudioMLP_Wrapper(
        sample_rate=16000,
        timestamp_embedding_size=embed_dim,
        scene_embedding_size=scene_dim,
        encoder_type=encoder_type,
        encoder_ckpt=model_file_path
    )
    
    assert isinstance(model, nn.Module)
    assert hasattr(model, "sample_rate")
    assert hasattr(model, "timestamp_embedding_size")
    assert hasattr(model, "scene_embedding_size")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model
    

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
    
    embeddings = model(audio) # (b, t, f)

    embed_t = model.scene_embedding_size / model.timestamp_embedding_size # 1024 / 64 = 16
    actual_t = embeddings.shape[1]

    if actual_t < embed_t:
        embeddings = F.pad(embeddings, (0, 0, 0, embed_t - actual_t), "constant", 0)
    elif embed_t > actual_t:
        p = embed_t - actual_t % embed_t
        embeddings = F.pad(embeddings, (0, 0, 0, p), "constant", 0)

    embeddings = rearrange(embeddings, "b (t n) f -> b t n f", t=embed_t) # (b, 16, n, 64)

    # reduce (b, 16, n, 64) to (b, 16, 64)
    embeddings = embeddings.mean(axis=2)
    
    # flatten in column major order
    return rearrange(embeddings, "b t f -> b (f t)")
