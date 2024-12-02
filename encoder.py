import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from toponetx.classes import SimplicialComplex
from tqdm import tqdm
import concurrent.futures
import heapq
import numpy as np
from itertools import combinations
import math
import random
from rectifier import ConstraintMatrices, enforce_constraints, RectifiedProbs
from complex_builder import SparseSimplicialMatrices, build_sparse_matrices

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class HardConcrete(nn.Module):
    def __init__(self, 
                 beta=1.0, 
                 init_gamma=-0.1, 
                 init_zeta=1.1, 
                 fix_params=False, 
                 loc_bias=0.5,
                 eps = 1e-7):
        super().__init__()
        
        self.eps = eps
        # Stretching parameters
        if fix_params:
            self.temp = beta
            self.gamma = init_gamma
            self.zeta = init_zeta
            self.loc_bias = loc_bias
        else:
            self.temp = nn.Parameter(torch.ones(1) * beta)
            self.gamma = nn.Parameter(torch.ones(1) * init_gamma)
            self.zeta = nn.Parameter(torch.ones(1) * init_zeta)
            self.loc_bias = nn.Parameter(torch.ones(1) * loc_bias)
            
        self.register_buffer('uniform', torch.zeros(1))

    def forward(self, input_scores):
        if self.training:
            self.uniform.uniform_(self.eps, 1 - self.eps)
            noise = torch.log(self.uniform) - torch.log(1 - self.uniform)
            scores = input_scores + self.loc_bias + noise
        else:
            scores = input_scores + self.loc_bias

        concrete = torch.sigmoid(scores / self.temp)
        
        # Ensure zeta > gamma for valid stretching
        gamma = -F.softplus(-self.gamma)
        zeta = gamma + F.softplus(self.zeta)  # Ensures zeta > gamma by at least 1.2
        
        stretched = gamma + (zeta - gamma) * concrete
        clamped = torch.clamp(stretched, 0, 1)
        return stretched + (clamped - stretched).detach()
    

class FrequencyBandModule(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=4):
        super().__init__()
        self.band_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.band_conv(x)
    

class AudioEncoder(nn.Module):
    def __init__(self, num_vertices, num_bands=16, embedding_dim=128):
        super(AudioEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.num_edges = math.comb(self.num_vertices, 2)
        self.num_triangles = math.comb(self.num_vertices, 3)
        self.num_tetra = math.comb(self.num_vertices, 4)
        self.total_simplices = self.num_vertices + self.num_edges + self.num_triangles + self.num_tetra
        # self.sc = SimplicialComplex()
        self.seed = 511990
        self.embedding_dim = embedding_dim

        # Vertex embeddings
        self.vertex_embeddings = nn.Embedding(num_vertices, embedding_dim)

        # Linear layers for projecting embeddings
        self.edge_projection = nn.Linear(2 * embedding_dim, embedding_dim)
        self.triangle_projection = nn.Linear(3 * embedding_dim, embedding_dim)
        self.tetra_projection = nn.Linear(4 * embedding_dim, embedding_dim)

        # Create constraint matrices for rectification
        self.constraints = ConstraintMatrices.create(num_vertices)
        
        # Process each frequency band independently
        out_channels = 16
        self.band_processors = nn.ModuleList([
            FrequencyBandModule(in_channels=1, out_channels=out_channels) 
            for _ in range(num_bands)
        ])

        self.cross_band = nn.Sequential(
            nn.Conv1d(out_channels * num_bands, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )

        self.temporal_reduction = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(),
            # Now temporal dimension is ~500
            nn.AdaptiveAvgPool1d(output_size=16)  # Less drastic pooling
        )

        self.mlp = nn.Sequential(
            nn.Linear(128 * 16, 512),  # 128 channels * 16 temporal positions
            nn.LeakyReLU(),
            nn.Linear(512, self.total_simplices)
        )
        
        # Hard concrete layers
        self.concrete = HardConcrete()
        # self.vertex_concrete = HardConcrete()
        # self.edge_concrete = HardConcrete()
        # self.triangle_concrete = HardConcrete()
        # self.tetra_concrete = HardConcrete()

    def forward(self, x):
        # x shape: [batch, 16, 64000]
        
        # Process each frequency band independently
        band_features = []
        for i, band_processor in enumerate(self.band_processors):
            band = x[:, i:i+1]  # Get single frequency band [batch, 1, 64000]
            band_features.append(band_processor(band))
        
        # Concatenate all band features
        x = torch.cat(band_features, dim=1)  # [batch, 16*16, 16000] (due to two stride-2 convs)
        
        # Cross-band integration
        x = self.cross_band(x)  # [batch, 128, 8000] (due to stride-2)
        
        # Temporal reduction
        x = self.temporal_reduction(x)  # [batch, 128, 16]
        
        # Flatten and pass through MLP
        x = x.flatten(1)  # [batch, 128*16]
        logits = self.mlp(x)  # [batch, total_simplices]
        
        # Apply Hard Concrete for initial sparsity
        probs = self.concrete(logits)   

        # Split probs
        vertex_probs = probs[:self.num_vertices]
        edge_probs = probs[self.num_vertices:self.num_vertices + self.num_edges]
        triangle_probs = probs[self.num_vertices + self.num_edges:self.num_vertices + self.num_edges + self.num_triangles]
        tetra_probs = probs[-self.num_tetra:]
        
        # Rectify probabilities to ensure valid simplicial complex
        rectified = enforce_constraints(
            vertex_probs,
            edge_probs,
            triangle_probs,
            tetra_probs,
            self.constraints
        )

        # Apply binary entropy loss to push probabilities towards 0 or 1
        # Calculate binary entropy loss for each type of simplex using list comprehension
        binary_entropy = sum(
            -(p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10))
            for p in [rectified.vertices, rectified.edges, rectified.triangles, rectified.tetra]
        )
        entropy_loss = binary_entropy.mean()

        complex_matrices = build_sparse_matrices(rectified, self.constraints)

        # Create vertex embeddings

        vertex_indices = torch.arange(self.num_vertices, device=x.device)
        # vertex_indices = vertex_probs.nonzero().squeeze(1)
        vertex_embeds = self.vertex_embeddings(vertex_indices)

        # Use pre-computed indices from constraints
        edge_indices = self.constraints.indices.edges
        triangle_indices = self.constraints.indices.triangles
        tetra_indices = self.constraints.indices.tetra

        # Project edge embeddings
        edge_embeds = self.edge_projection(vertex_embeds[edge_indices].view(-1, 2 * self.embedding_dim))

        # Project triangle embeddings
        triangle_embeds = self.triangle_projection(edge_embeds[triangle_indices].view(-1, 3 * self.embedding_dim))

        # Project tetrahedron embeddings
        tetra_embeds = self.tetra_projection(triangle_embeds[tetra_indices].view(-1, 4 * self.embedding_dim))

        # Concatenate all embeddings
        embeddings = {
            0 : vertex_embeds, 
            1 : edge_embeds, 
            2 : triangle_embeds, 
            3 : tetra_embeds
        }

        # Return embeddings as part of the output
        return embeddings, complex_matrices, entropy_loss

        # vertex_logits = logits[:, :self.num_vertices]
        # edge_logits = logits[:, self.num_vertices:self.num_vertices + self.num_edges]
        # triangle_logits = logits[:, self.num_vertices + self.num_edges:self.num_vertices + self.num_edges + self.num_triangles]
        # tetra_logits = logits[:, -self.num_tetra:]
        
        # # Apply Hard Concrete for initial sparsity
        # vertex_probs = self.vertex_concrete(vertex_logits)
        # edge_probs = self.edge_concrete(edge_logits)
        # triangle_probs = self.triangle_concrete(triangle_logits)
        # tetra_probs = self.tetra_concrete(tetra_logits)
        
        # return rectified.vertices, rectified.edges, rectified.triangles, rectified.tetra
    

    def count_parameters(self):
        """Count and print parameters for each component of the model"""
        
        # FrequencyBandModule params (per band)
        band_params = sum(p.numel() for p in self.band_processors[0].parameters())
        total_band_params = band_params * len(self.band_processors)
        
        # Cross-band params
        cross_band_params = sum(p.numel() for p in self.cross_band.parameters())
        
        # Temporal reduction params
        temporal_params = sum(p.numel() for p in self.temporal_reduction.parameters())
        
        # MLP params
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        
        # Hard Concrete params
        # concrete_params = sum(sum(p.numel() for p in module.parameters()) 
        #                 for module in [
        #                     self.vertex_concrete,
        #                     self.edge_concrete,
        #                     self.triangle_concrete,
        #                     self.tetra_concrete
        #                 ])
        concrete_params = sum(p.numel() for p in self.concrete.parameters())
        
        print("\nParameter Count Breakdown:")
        print(f"Frequency Band Processors: {total_band_params:,} params")
        print(f"  (Per band: {band_params:,} params × {len(self.band_processors)} bands)")
        print(f"Cross-band Integration: {cross_band_params:,} params")
        print(f"Temporal Reduction: {temporal_params:,} params")
        print(f"MLP: {mlp_params:,} params")
        print(f"Hard Concrete: {concrete_params:,} params")
        print(f"Total: {self.num_params():,} params")
        
        # Detailed breakdown of a single FrequencyBandModule
        print("\nFrequencyBandModule Breakdown:")
        for name, param in self.band_processors[0].named_parameters():
            print(f"{name}: {param.numel():,} params")
        
        # Detailed breakdown of MLP
        print("\nMLP Breakdown:")
        for name, param in self.mlp.named_parameters():
            print(f"{name}: {param.numel():,} params")

    def num_params(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def test_hard_concrete():
    # Create logits with a distribution more like what the network might output
    logits = torch.randn(1000) * 2  # Normal distribution with std=2
    print(f"\nLogits stats:")
    print(f"Mean: {logits.mean():.3f}")
    print(f"Std: {logits.std():.3f}")
    print(f"Min: {logits.min():.3f}")
    print(f"Max: {logits.max():.3f}")

    # Test different biases through sigmoid
    print("\nJust sigmoid with different biases:")
    for bias in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        probs = torch.sigmoid(logits + bias)
        print(f"Bias {bias:4.1f}: avg prob = {probs.mean():.3f}")

    # Test full Hard Concrete
    print("\nFull Hard Concrete distribution:")
    hc = HardConcrete(
        beta=0.6,
        init_gamma=-0.1,
        init_zeta=1.1,
        loc_bias=2.0
    )
    
    # Test in training mode (with noise)
    hc.train()
    print("\nTraining mode (with noise):")
    for _ in range(5):  # Multiple runs to see variance
        out = hc(logits)
        zeros = (out == 0.0).float().mean()
        ones = (out == 1.0).float().mean()
        middle = ((out > 0.0) & (out < 1.0)).float().mean()
        print(f"Run {_+1}:")
        print(f"  Zeros: {zeros:.3f}")
        print(f"  Ones: {ones:.3f}")
        print(f"  In (0,1): {middle:.3f}")
        print(f"  Mean: {out.mean():.3f}")

    # Test in eval mode (deterministic)
    hc.eval()
    print("\nEval mode (no noise):")
    out = hc(logits)
    zeros = (out == 0.0).float().mean()
    ones = (out == 1.0).float().mean()
    middle = ((out > 0.0) & (out < 1.0)).float().mean()
    print(f"Zeros: {zeros:.3f}")
    print(f"Ones: {ones:.3f}")
    print(f"In (0,1): {middle:.3f}")
    print(f"Mean: {out.mean():.3f}")

def test_simple_complex():
    # Define 6 vertices
    n_vertices = 6
    
    # Create binary probabilities for a simple complex
    vertex_probs = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.float32)
    edge_probs = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)  # 15 edges for 6 vertices
    triangle_probs = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)  # 20 triangles
    tetra_probs = torch.tensor([1], dtype=torch.float32)  # At least one tetrahedron

    # Create constraint matrices
    matrices = ConstraintMatrices.create(n_vertices)

    # Create rectified probabilities
    rectified_probs = enforce_constraints(
        vertex_probs=vertex_probs,
        edge_probs=edge_probs,
        triangle_probs=triangle_probs,
        tetra_probs=tetra_probs,
        matrices=matrices
    )

    print("rectified_probs.vertices")
    print(rectified_probs.vertices)
    print("rectified_probs.edges")
    print(rectified_probs.edges)
    print("rectified_probs.triangles")
    print(rectified_probs.triangles)
    print("rectified_probs.tetra")
    print(rectified_probs.tetra)

    # Build sparse matrices
    sparse_matrices = build_sparse_matrices(rectified_probs, matrices)

    # Print adjacency and incidence matrices
    print("Vertex Adjacency Matrix:")
    print(sparse_matrices.adjacencies[0].to_dense())

    print("\nEdge Adjacency Matrix:")
    print(sparse_matrices.adjacencies[1].to_dense())

    print("\nTriangle Adjacency Matrix:")
    print(sparse_matrices.adjacencies[2].to_dense())

    print("\nTetrahedra Adjacency Matrix:")
    print(sparse_matrices.adjacencies[3].to_dense())

    print("\nVertex-Edge Incidence Matrix:")
    print(sparse_matrices.incidences[1].to_dense())

    print("\nEdge-Triangle Incidence Matrix:")
    print(sparse_matrices.incidences[2].to_dense())

    print("\nTriangle-Tetra Incidence Matrix:")
    print(sparse_matrices.incidences[3].to_dense())

if __name__ == "__main__":
    # model = AudioEncoder(num_vertices=20)
    # model.count_parameters()
    test_simple_complex()
    # test_hard_concrete()