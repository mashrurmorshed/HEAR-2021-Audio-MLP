import torch
from torch import nn
from typing import Tuple
from .models import AudioMLP


def load_model(model_file_path: str) -> nn.Module:
    """Loads model weights from provided path.

    Args:
        model_file_path (str): Provided checkpoint path.

    Returns:
        nn.Module: Model instance.
    """
    # ckpt = torch.load(model_file_path, map_location="cpu")
    model = AudioMLP()
    return model


def get_timestamp_embeddings(audio: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns embeddings at regular intervals centered at timestamps, as well as the timestamps themselves.

    Args:
        audio (torch.Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embedding (torch.Tensor): A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps (torch.Tensor): A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """
    raise NotImplementedError


def get_scene_embeddings(audio: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Returns a single embedding for the entire audio clip.

    Args:
        audio (torch.Tensor): n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length. 
        model (nn.Module): Loaded model.

    Returns:
        embedding (torch.Tensor): A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    raise NotImplementedError