import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import AudioEncoder
from decoder import AudioDecoder
from rave.pqmf import PQMF
import random

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class AudioAutoencoder(nn.Module):
    def __init__(self, num_vertices, num_bands=16, decoder_hidden_dim=256):
        super(AudioAutoencoder, self).__init__()
        
        self.encoder = AudioEncoder(num_vertices, num_bands=num_bands, embedding_dim=decoder_hidden_dim)
        self.decoder = AudioDecoder(decoder_hidden_dim=decoder_hidden_dim)
        self.test = False
        # self.encoder_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.encoder_device = torch.device("cpu")
        # self.sccn_device = torch.device("cpu")
        self.pqmf = PQMF(100, num_bands, polyphase=True)
        self.seed = 511990
        self.sc = None

    def forward(self, x):
        x = self.pqmf.forward(x)
        x = x.squeeze(0)
        desired_length = x.shape[1]
        # x = x.to(self.encoder_device).squeeze(0)
        feature_embeddings, complex_matrices, loss_component = self.encoder(x)

        # self.sc = sc
        output = self.decoder(feature_embeddings, complex_matrices, desired_length)
        # output = output.to(self.sccn_device)
        output = self.pqmf.inverse(output)
        # output = output.to(self.encoder_device)

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
        