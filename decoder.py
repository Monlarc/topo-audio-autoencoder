import torch
import torch.nn as nn
import torch.nn.functional as F
from TopoModelX.topomodelx.nn.simplicial.sccn import SCCN
import random
torch.autograd.set_detect_anomaly(True)
from torch.profiler import profile, record_function, ProfilerActivity
import time
from custom_sccn import GradientSCCN, JumpingKnowledgeSCCN

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class AudioDecoder(nn.Module):
    def __init__(self, 
                 sccn_hidden_dim=64,
                 initial_sequence_length=250,
                 output_channels=16):
        super().__init__()
        self.sccn = JumpingKnowledgeSCCN(
            channels=sccn_hidden_dim, 
            max_rank=3, 
            n_layers=4, 
            update_func='gelu'
        )
        self.initial_sequence_length = initial_sequence_length
        
        # Scale down initial query values
        self.query_sequence = nn.Parameter(
            torch.randn(1, initial_sequence_length, sccn_hidden_dim) * 0.02
        )
        
        # Add layer norm before and after attention
        self.pre_attention_norm = nn.LayerNorm(sccn_hidden_dim)
        self.post_attention_norm = nn.LayerNorm(sccn_hidden_dim)
        
        # Cross-attention with gradient scaling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=sccn_hidden_dim,
            num_heads=16,
            batch_first=True,
            dropout=0.1  # Add some dropout
        )
        
        # Scale attention outputs to help with gradient flow
        self.attention_scale = nn.Parameter(torch.ones(1))
        
        # Add pre-scaling for queries and keys
        self.query_proj = nn.Linear(sccn_hidden_dim, sccn_hidden_dim)
        self.key_proj = nn.Linear(sccn_hidden_dim, sccn_hidden_dim)
        self.value_proj = nn.Linear(sccn_hidden_dim, sccn_hidden_dim)
        
        # Modify upsampling blocks for better gradient flow
        self.upsample_blocks = nn.ModuleList([
            # 250 -> 500
            nn.Sequential(
                nn.ConvTranspose1d(sccn_hidden_dim, sccn_hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim),  # Group norm instead of batch norm
                nn.GELU(),  # GELU instead of LeakyReLU
                nn.Conv1d(sccn_hidden_dim, sccn_hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim),
                nn.GELU(),
            ),
            # 500 -> 1000
            nn.Sequential(
                nn.ConvTranspose1d(sccn_hidden_dim, sccn_hidden_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(sccn_hidden_dim // 2, sccn_hidden_dim // 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim // 2),
                nn.GELU(),
            ),
            # 1000 -> 2000
            nn.Sequential(
                nn.ConvTranspose1d(sccn_hidden_dim // 2, sccn_hidden_dim // 4, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim // 4),
                nn.GELU(),
                nn.Conv1d(sccn_hidden_dim // 4, sccn_hidden_dim // 4, kernel_size=3, padding=1),
                nn.GroupNorm(8, sccn_hidden_dim // 4),
                nn.GELU(),
            ),
            # 2000 -> 4000
            nn.Sequential(
                nn.ConvTranspose1d(sccn_hidden_dim // 4, output_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, output_channels),
                nn.GELU(),
                nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            )
        ])
        
        # Add residual projections for each upsampling block
        self.residual_projections = nn.ModuleList([
            nn.Conv1d(sccn_hidden_dim, sccn_hidden_dim, kernel_size=1),
            nn.Conv1d(sccn_hidden_dim, sccn_hidden_dim // 2, kernel_size=1),
            nn.Conv1d(sccn_hidden_dim // 2, sccn_hidden_dim // 4, kernel_size=1),
            nn.Conv1d(sccn_hidden_dim // 4, output_channels, kernel_size=1)
        ])
        
        # Add scaling factors for residual connections
        self.residual_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(4)
        ])
        
        # Initialize weights with smaller values
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, decoder_input_features, complex_matrices, desired_length):
        # Process through SCCN
        output = self.sccn(decoder_input_features, complex_matrices.incidences, complex_matrices.adjacencies)
        
        # Get features from all ranks and concatenate
        all_features = []
        for rank in range(4):  # 0 to 3
            rank_features = output[f'rank_{rank}'].to_dense()
            all_features.append(rank_features)
        
        # Concatenate along feature dimension
        combined_features = torch.cat(all_features, dim=0)  # [total_simplices, hidden_dim]
        
        # Add batch dimension to active features
        combined_features = combined_features.unsqueeze(0)  # [1, total_simplices, hidden_dim]
        
        # Normalize inputs
        combined_features = self.pre_attention_norm(combined_features)
        query = self.query_proj(self.pre_attention_norm(self.query_sequence))
        keys = self.key_proj(self.pre_attention_norm(combined_features))
        values = self.value_proj(combined_features)
        
        # Scale attention outputs
        attn_out, _ = self.cross_attention(
            query=query,
            key=keys,
            value=values
        )
        attn_out = attn_out * self.attention_scale
        
        # Gradient-friendly residual connection
        x = query + F.gelu(attn_out)  # Using GELU instead of direct addition
        x = self.post_attention_norm(x)
        
        # Prepare for convolutional layers
        x = x.transpose(1, 2)
        
        # Progressive upsampling with scaled residual connections
        for block, proj, scale in zip(self.upsample_blocks, self.residual_projections, self.residual_scales):
            identity = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            identity = proj(identity) * scale  # Scale residual connection
            x = block(x)
            x = x + identity
            
        return x

    
