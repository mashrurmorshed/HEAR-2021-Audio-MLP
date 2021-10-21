import torch
from torch import nn

class AudioMLP(nn.Module):
    """Shell for model."""
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.scene_embedding_size = 1024
        self.timestamp_embedding_size = 1024

    def forward(self, input):
        raise NotImplementedError
