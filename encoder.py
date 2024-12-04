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
torch.autograd.set_detect_anomaly(True)
from torch.profiler import profile, record_function, ProfilerActivity

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g

class GumbelBernoulli(nn.Module):
    def __init__(self, start_temp=5.0, min_temp=0.1):
        super().__init__()
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.current_temp = start_temp
        
    def forward(self, logits):
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.current_temp
            y_soft = F.softplus(gumbels) / (F.softplus(gumbels) + F.softplus(-gumbels))
            
            # Sample from Bernoulli
            y_hard = torch.bernoulli(y_soft)
            
            return y_soft + (y_hard - y_soft).detach()
        else:
            logits = logits / self.current_temp
            return (F.softplus(logits) / (F.softplus(logits) + F.softplus(-logits)) > 0.5).float()

    def set_temperature(self, temp):
        self.current_temp = temp


class FrequencyBandModule(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=16):
        super().__init__()
        self.band_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.band_conv(x)
    

class AudioEncoder(nn.Module):
    def __init__(self, num_vertices, num_bands=16, embedding_dim=128, dropout=0.1):
        super().__init__()
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
        self.band_processors = nn.ModuleList([
            FrequencyBandModule(in_channels=1, out_channels=16) 
            for _ in range(num_bands)
        ])

        self.cross_band = nn.Sequential(
            nn.Conv1d(16 * num_bands, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )

        post_temporal_reduction = 64
        self.temporal_reduction = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=post_temporal_reduction)
        )

        self.mlp = nn.Sequential(
            nn.Linear(128 * post_temporal_reduction, 1024),  # 128 channels * 32 temporal positions
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.total_simplices)
        )
        
        # Replace Hard Concrete with Gumbel-Bernoulli
        self.gumbel = GumbelBernoulli()
        self.mlp_output = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Process each frequency band independently
        band_features = []
        for i, band_processor in enumerate(self.band_processors):
            band = x[:, i:i+1]
            processed = band_processor(band)
            band_features.append(processed)
        x = torch.cat(band_features, dim=1)
        x = self.dropout(x)
        
        # Cross-band integration and temporal reduction
        x = self.cross_band(x)
        # x = self.dropout(x)
        x = self.temporal_reduction(x)
        x = x.flatten(1)
        # x = self.dropout(x)
        logits = self.mlp(x).squeeze(0)
        
        # Add bias to vertex logits to encourage some active vertices
        vertex_logits = logits[:self.num_vertices] + 5.0  # Bias towards active vertices
        edge_logits = logits[self.num_vertices:self.num_vertices + self.num_edges]
        triangle_logits = logits[self.num_vertices + self.num_edges:
                                self.num_vertices + self.num_edges + self.num_triangles]
        tetra_logits = logits[-self.num_tetra:]
        
        # Apply Gumbel-Bernoulli to each set
        # while True:
        #     vertex_probs = self.gumbel(vertex_logits)
        #     if torch.sum(vertex_probs) > 2:
        #         break
        #     vertex_logits = vertex_logits + torch.randn_like(vertex_logits)
        vertex_probs = self.gumbel(vertex_logits) 
        edge_probs = self.gumbel(edge_logits)
        triangle_probs = self.gumbel(triangle_logits)
        tetra_probs = self.gumbel(tetra_logits)
        
        # Build simplicial complex
        rectified = enforce_constraints(
            vertex_probs, edge_probs, triangle_probs, tetra_probs, self.constraints
        )

        # print(f"rectified.vertices: {rectified.vertices}")
        if torch.sum(rectified.vertices) == 0:
            # print(f"rectified.vertices: {rectified.vertices}")
            # print("Warning: No simplices found in rectified complex")
            return None, None, None
        
        print(f"Number of vertices: {len(rectified.vertices.nonzero())}")
        print(f"Number of edges: {len(rectified.edges.nonzero())}")
        print(f"Number of triangles: {len(rectified.triangles.nonzero())}")
        print(f"Number of tetrahedra: {len(rectified.tetra.nonzero())}")
        
        # Faster vectorized implementation of entropy loss
        all_probs = torch.cat([rectified.vertices, rectified.edges, rectified.triangles, rectified.tetra])
        entropy_loss = -(all_probs * torch.log(all_probs + 1e-10) + 
                        (1 - all_probs) * torch.log(1 - all_probs + 1e-10)).mean()
        
        self.complex_matrices = build_sparse_matrices(rectified, self.constraints)
        
        # Generate embeddings
        vertex_indices = torch.arange(self.num_vertices, device=x.device)
        vertex_embeds = self.vertex_embeddings(vertex_indices)
        
        edge_indices = self.constraints.indices.edges
        triangle_indices = self.constraints.indices.triangles
        tetra_indices = self.constraints.indices.tetra
        
        edge_embeds = self.edge_projection(vertex_embeds[edge_indices].view(-1, 2 * self.embedding_dim))
        triangle_embeds = self.triangle_projection(edge_embeds[triangle_indices].view(-1, 3 * self.embedding_dim))
        tetra_embeds = self.tetra_projection(triangle_embeds[tetra_indices].view(-1, 4 * self.embedding_dim))
        
        embeddings = {
            'rank_0': vertex_embeds,
            'rank_1': edge_embeds,
            'rank_2': triangle_embeds,
            'rank_3': tetra_embeds
        }
        
        self.mlp_output = logits
        
        return embeddings, self.complex_matrices, entropy_loss

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
        
        # Gumbel-Bernoulli params
        gumbel_params = sum(p.numel() for p in self.gumbel.parameters())
        
        print("\nParameter Count Breakdown:")
        print(f"Frequency Band Processors: {total_band_params:,} params")
        print(f"  (Per band: {band_params:,} params × {len(self.band_processors)} bands)")
        print(f"Cross-band Integration: {cross_band_params:,} params")
        print(f"Temporal Reduction: {temporal_params:,} params")
        print(f"MLP: {mlp_params:,} params")
        print(f"Gumbel-Bernoulli: {gumbel_params:,} params")
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
    
    def get_last_layer_output(self):
        return self.mlp_output

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
        beta=1.0,
        init_gamma=-1.0,
        init_zeta=1.1,
        loc_bias=0.0
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
    # test_simple_complex()
    test_hard_concrete()