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
        self.sccn = GradientSCCN(
            channels=sccn_hidden_dim, 
            max_rank=3, 
            n_layers=6, 
            update_func='gelu'
        )
        self.initial_sequence_length = initial_sequence_length
        
        # Process vertex features with gradient scaling
        self.vertex_to_query = nn.Sequential(
            nn.Linear(sccn_hidden_dim, sccn_hidden_dim * 2),
            nn.LayerNorm(sccn_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(sccn_hidden_dim * 2, sccn_hidden_dim),
            nn.LayerNorm(sccn_hidden_dim),
            nn.GELU(),
        )
        
        # Temporal convolution with gradient scaling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(sccn_hidden_dim, sccn_hidden_dim, kernel_size=3, padding=1, groups=8),  # Depthwise conv
            nn.GroupNorm(8, sccn_hidden_dim),
            nn.GELU(),
            nn.Conv1d(sccn_hidden_dim, sccn_hidden_dim, kernel_size=3, padding=1, groups=8),
            nn.GroupNorm(8, sccn_hidden_dim),
            nn.GELU(),
        )
        
        # Layer norms for feature normalization
        self.pre_attention_norm = nn.LayerNorm(sccn_hidden_dim)
        self.post_attention_norm = nn.LayerNorm(sccn_hidden_dim)
        
        # Lightweight cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=sccn_hidden_dim,
            num_heads=4,  # Further reduce heads
            batch_first=True,
            dropout=0.0
        )
        
        # Smaller attention scale
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # Lightweight key/value projections using bottleneck architecture
        projection_dim = sccn_hidden_dim // 2  # Reduce dimensionality in middle layer
        self.key_proj = nn.Sequential(
            nn.Linear(sccn_hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, sccn_hidden_dim),
            nn.LayerNorm(sccn_hidden_dim)
        )
        self.value_proj = nn.Sequential(
            nn.Linear(sccn_hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, sccn_hidden_dim),
            nn.LayerNorm(sccn_hidden_dim)
        )
        
        # Progressive upsampling with gradient control
        channels = [sccn_hidden_dim, sccn_hidden_dim//2, sccn_hidden_dim//4, output_channels]
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(4):
            in_channels = channels[i]
            out_channels = channels[min(i+1, len(channels)-1)]
            scale_factor = 1.0 / (2 ** (i + 1))  # Progressive scaling
            
            block = nn.Sequential(
                # Use bilinear upsampling instead of transposed conv
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                # Depthwise separable convolutions
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                # Scale output
                ScaleLayer(scale_factor)
            )
            self.upsample_blocks.append(block)
        
        # Initialize weights carefully
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, feature_embeddings, complex_matrices, desired_length):
        """
        Forward pass of the decoder.
        Args:
            feature_embeddings: Dictionary of embeddings for each rank
            complex_matrices: Sparse matrices representing the simplicial complex
            desired_length: Desired length of the output audio
        """
        # Process through SCCN
        output = self.sccn(feature_embeddings, complex_matrices.incidences, complex_matrices.adjacencies)
        
        # Get vertex features and process them into query sequence
        vertex_features = output['rank_0'].to_dense() * 0.1  # Scale down initial features
        vertex_features = self.vertex_to_query(vertex_features)
        
        # Convert to sequence via temporal convolution
        vertex_features = vertex_features.transpose(0, 1).unsqueeze(0)
        query = self.temporal_conv(vertex_features)
        
        # Interpolate to desired sequence length
        query = F.interpolate(query, size=self.initial_sequence_length, mode='linear', align_corners=False)
        query = query.transpose(1, 2)
        
        # Get features from higher ranks for attention
        higher_rank_features = []
        for rank in range(1, 4):
            rank_key = f'rank_{rank}'
            if rank_key in output and output[rank_key] is not None:
                rank_features = output[rank_key].to_dense() * 0.1  # Scale down higher rank features
                higher_rank_features.append(rank_features)
        
        # Concatenate features for attention
        combined_features = torch.cat(higher_rank_features, dim=0).unsqueeze(0)
        
        # Normalize and project features
        combined_features = self.pre_attention_norm(combined_features)
        query = self.pre_attention_norm(query)
        keys = self.key_proj(combined_features)
        values = self.value_proj(combined_features)
        
        # Attention with scaled output
        attn_out, _ = self.cross_attention(query=query, key=keys, value=values)
        attn_out = attn_out * self.attention_scale
        
        # Residual connection with normalization
        x = query + F.gelu(attn_out)
        x = self.post_attention_norm(x)
        
        # Prepare for upsampling
        x = x.transpose(1, 2)
        
        # Progressive upsampling with gradient scaling
        for block in self.upsample_blocks:
            x = block(x)
            
        return x

class ScaleLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return x * self.scale_factor

    
