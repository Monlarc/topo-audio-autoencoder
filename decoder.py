from utils import PriorityQueue
import torch
import torch.nn as nn
import torch.nn.functional as F
from TopoModelX.topomodelx.nn.simplicial.sccn import SCCN
import random

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class AudioDecoder(nn.Module):
    def __init__(self, decoder_hidden_dim=256):
        super(AudioDecoder, self).__init__()
        # self.mps_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.mps_device = torch.device("cpu")
        # self.sccn_device = torch.device('cpu')
        self.sccn = SCCN(channels=decoder_hidden_dim, max_rank=3, n_layers=6, update_func='sigmoid')
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=decoder_hidden_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=2
        )

        self.out_conv1 = nn.Conv1d(in_channels=decoder_hidden_dim, out_channels=64, kernel_size=7, stride=1, padding='same')
        self.out_conv2 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=31, stride=1, padding='same')
        self.conv_tranpose = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=4, stride=4)
        self.seed = 511990

    def forward(self, decoder_input_features, complex_matrices, desired_length):
        # for tensors in decoder_input_features.values():
        #     tensors.to(self.sccn_device)
        output = self.sccn(decoder_input_features, complex_matrices.incidences, complex_matrices.adjacencies)

        features = [feature.to_dense() for feature in output.values()]
        
        decoder_tranformer_input = torch.cat(tuple(features), dim=0).unsqueeze(0)

        output_sequence = []
        output_size = 0
        while output_size < desired_length:

            output = self.transformer2(decoder_tranformer_input)  # Assuming `memory` is handled inside if needed
            output_sequence.append(output)
            output_size  += output.shape[1]
            
            next_input = output[:, -1:, :]  # Take last timestep's output
            decoder_tranformer_input = torch.cat((decoder_tranformer_input, next_input), dim=1)

        output = torch.cat(output_sequence, dim=1)
        output = output[:,:desired_length,:].squeeze().transpose(0,1)
        output = self.out_conv1(output)
        output = F.relu(output)
        # output = self.out_conv2(output)
        output = self.conv_tranpose(output).unsqueeze(0)

        return output
    

