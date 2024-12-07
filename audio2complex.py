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
    def __init__(self, num_vertices, num_bands=16, sccn_hidden_dim=64,
                 min_active_vertices=8, max_active_vertices=20):
        super(AudioAutoencoder, self).__init__()
        
        self.encoder = AudioEncoder(
            num_vertices=num_vertices, 
            num_bands=num_bands, 
            embedding_dim=sccn_hidden_dim,
            min_active_vertices=min_active_vertices,
            max_active_vertices=max_active_vertices
        )
        self.decoder = AudioDecoder(
            sccn_hidden_dim=sccn_hidden_dim, 
            initial_sequence_length=250, 
            output_channels=num_bands
        )
        self.test = False
        self.pqmf = PQMF(100, num_bands, polyphase=True)
        self.seed = 511990
        self.sc = None

    def forward(self, x):
        # Process through PQMF
        out = self.pqmf.forward(x)
        desired_length = out.shape[2]
        
        # Encode
        feature_embeddings, complex_matrices, diversity_loss = self.encoder(out)
        if feature_embeddings is None or complex_matrices is None:
            return None, None

        # Decode
        output = self.decoder(feature_embeddings, complex_matrices, desired_length)
        output = self.pqmf.inverse(output)

        # Make sure input and output have same shape
        if output.shape != x.shape:
            output = output.view(x.shape)

        return output, diversity_loss
    

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
        