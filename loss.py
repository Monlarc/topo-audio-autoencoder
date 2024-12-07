from rave.core import AudioDistanceV1, MultiScaleSTFT, rearrange
from typing import Callable, Optional, Sequence, Union
import torch.nn as nn
import torch
import torchaudio
from torch import Generator


def set_seeds(seed):
    torch.manual_seed(seed)
    g = Generator(device='cpu')
    g.manual_seed(seed)
    return g

class AutoencoderLoss(AudioDistanceV1):
    def __init__(self,
                 binary_entropy_penalty=0.01,
                 min_entropy_penalty=0.01,
                 complexity_penalty=0.1) -> None:
        def create_multiscale_stft():
            return MultiScaleSTFT(scales=[2048, 1024, 512, 256, 128], magnitude=True, sample_rate=16000)
        super().__init__(create_multiscale_stft, 1e-7)
        self.binary_entropy_penalty = binary_entropy_penalty
        self.min_entropy_penalty = min_entropy_penalty
        self.complexity_penalty = complexity_penalty

    def forward(self, x: torch.Tensor, y: torch.Tensor, diversity_loss: float):
        # Reconstruction loss from parent class
        result = super().forward(x, y)
        spectral_loss = result['spectral_distance']

        entropy_loss = diversity_loss['binary_entropy']
        vertex_loss = diversity_loss['diversity']
        
        # Combine all loss components with their respective weights
        # 1. Spectral reconstruction loss
        # 2. Binary entropy loss to maintain stochastic behavior
        # 3. Diversity loss to prevent mode collapse
        # 4. Minimum entropy penalty to ensure some randomness
        total_loss = (
            spectral_loss + 
            self.binary_entropy_penalty * entropy_loss +
            self.complexity_penalty * vertex_loss
        )
        
        # Store individual loss components for logging
        self.loss_components = {
            'spectral_loss': spectral_loss.item(),
            'binary_entropy_loss': entropy_loss.item(),
            'diversity_loss': vertex_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss