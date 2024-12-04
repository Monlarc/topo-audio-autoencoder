import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import AudioEncoder
from decoder import AudioDecoder
from rave.pqmf import PQMF
import random
torch.autograd.set_detect_anomaly(True)

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class AudioAutoencoder(nn.Module):
    def __init__(self, num_vertices, num_bands=16, sccn_hidden_dim=64):
        super(AudioAutoencoder, self).__init__()
        
        self.encoder = AudioEncoder(num_vertices, num_bands=num_bands, embedding_dim=sccn_hidden_dim)
        self.decoder = AudioDecoder(sccn_hidden_dim=sccn_hidden_dim, initial_sequence_length=250, output_channels=num_bands)
        self.test = False
        # self.encoder_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.encoder_device = torch.device("cpu")
        # self.sccn_device = torch.device("cpu")
        self.pqmf = PQMF(100, num_bands, polyphase=True)
        self.seed = 511990
        self.sc = None

    def forward(self, x):
        # print(f"Input to autoencoder shape: {x.shape}")
        out = self.pqmf.forward(x)
        # print(f"After PQMF shape: {out.shape}")
        desired_length = out.shape[2]
        # print(f"Desired length: {desired_length}")
        
        feature_embeddings, complex_matrices, loss_component = self.encoder(out)
        if feature_embeddings is None or complex_matrices is None:
            return None, None

        # self.sc = sc
        output = self.decoder(feature_embeddings, complex_matrices, desired_length)
        # print(f"Before PQMF inverse shape: {output.shape}")
        output = self.pqmf.inverse(output)
        # print(f"Final output shape: {output.shape}")

        # Make sure input and output have same shape
        if output.shape != x.shape:
            # print(f"Shape mismatch - Input: {x.shape}, Output: {output.shape}")
            output = output.view(x.shape)
            # print(f"Output shape after view: {output.shape}")

        return output, loss_component
    

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def reset_weights(self, init_as_start=True):
        if init_as_start:
            set_seeds(self.seed)
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def get_complex(self):
        return self.sc
        