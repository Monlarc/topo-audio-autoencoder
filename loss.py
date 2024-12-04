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
                 binary_entropy_penalty=0.01) -> None:
        def create_multiscale_stft():
            return MultiScaleSTFT(scales=[2048, 1024, 512, 256, 128], magnitude=True, sample_rate=16000)
        super().__init__(create_multiscale_stft, 1e-7)
        self.binary_entropy_penalty = binary_entropy_penalty

    def forward(self, x: torch.Tensor, y: torch.Tensor, binary_entropy_loss):
        result = super().forward(x, y)

        loss = result['spectral_distance'] + self.binary_entropy_penalty * binary_entropy_loss

        return loss