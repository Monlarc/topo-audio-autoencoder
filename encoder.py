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

class BinaryGumbel(nn.Module):
    def __init__(self, start_temp=1.0, min_temp=0.01):
        super().__init__()
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.current_temp = start_temp
        
    def forward(self, logits):
        logits_pair = torch.stack([logits, 1 - logits])
        if self.training:
            gumbels = -torch.empty_like(logits_pair).exponential_().log()
            gumbel_logits = (logits_pair + gumbels) / self.current_temp

            probs = F.softmax(gumbel_logits, dim=0)[0]

            return probs
        
        else:
            logits = logits / self.current_temp
            probs = F.softmax(logits, dim=0)[0]
                
            return (probs > 0.5).float()
            
    def set_temperature(self, temp):
        if temp < self.min_temp:
            self.current_temp = self.min_temp
        else:
            self.current_temp = temp


class FrequencyBandModule(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=16):
        super().__init__()
        self.band_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=15, stride=2, padding=7),
            nn.InstanceNorm1d(hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=15, stride=2, padding=7),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.band_conv(x)
    

class AudioEncoder(nn.Module):
    def __init__(
            self, 
            num_vertices,
            num_bands=16, 
            embedding_dim=128, 
            dropout=0.1,
            min_active_vertices=8, 
            max_active_vertices=16, 
            temp=5.0, 
            min_temp=0.1,
            hard=False
        ):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_edges = math.comb(self.num_vertices, 2)
        self.num_triangles = math.comb(self.num_vertices, 3)
        self.num_tetra = math.comb(self.num_vertices, 4)
        self.total_simplices = self.num_vertices + self.num_edges + self.num_triangles + self.num_tetra
        self.seed = 511990
        self.embedding_dim = embedding_dim
        self.min_active_vertices = min_active_vertices
        self.max_active_vertices = max_active_vertices
        self.num_bands = num_bands
        self.constraints = ConstraintMatrices.create(num_vertices)
        self.active_simplices = None
        self.hard = hard

        if hard:
            self.gumbel = BinaryGumbel(start_temp=1.0, min_temp=0.01)

        # Process each frequency band with progressive dimension changes
        self.band_processors = nn.ModuleList([
            nn.Sequential(
                # Initial feature extraction
                nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(2, 8),
                nn.GELU(),
                # First temporal reduction
                nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                # Second temporal reduction
                nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(4, 16),
                nn.GELU(),
            ) for _ in range(num_bands)
        ])

        # Simple skip connection with matched dimensions
        self.skip_maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))  # Learnable weight for skip connection

        # Cross-band processing with matching temporal reduction
        self.cross_band = nn.Sequential(
            # First merge: 256 -> 192
            nn.Conv1d(num_bands * 16, 192, kernel_size=5, padding=2, groups=4),
            nn.GroupNorm(12, 192),
            nn.GELU(),
            # Second merge: 192 -> 128
            nn.Conv1d(192, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )

        # Progressive temporal reduction
        self.temporal_reduction = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=4, padding=3, groups=8),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, groups=8),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # Final reduction
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        # Progressive MLP with residual connections
        self.to_simplices = nn.Sequential(
            # First reduction
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            # Second reduction
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            # Final projection
            nn.Linear(1024, self.total_simplices)
        )

        self.vertex_bias = nn.Parameter(torch.ones(1) * 2.0)
        self.edge_bias = nn.Parameter(torch.ones(1))
        self.triangle_bias = nn.Parameter(torch.ones(1))
        self.tetra_bias = nn.Parameter(torch.ones(1) * 1.5)
        
        self.gumbel = BinaryGumbel()
        self.dropout = nn.Dropout(dropout)
        self.temp = temp
        self.min_temp = min_temp

        self.vertex_embeddings = nn.Sequential(
            nn.Embedding(num_vertices, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.edge_embeddings = nn.Sequential(
            nn.Embedding(self.num_edges, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.triangle_embeddings = nn.Sequential(
            nn.Embedding(self.num_triangles, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.tetra_embeddings = nn.Sequential(
            nn.Embedding(self.num_tetra, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.constraints = ConstraintMatrices.create(num_vertices)

    def compute_vertex_penalty(self, vertex_probs):
        vertex_count = vertex_probs.sum()
        vertex_penalty = F.relu(self.min_active_vertices - vertex_count) + \
                        F.relu(vertex_count - self.max_active_vertices)
        return vertex_penalty
    
    def compute_entropy_loss(self, vertex_probs, edge_probs, triangle_probs, tetra_probs):
        # Compute average activation for each simplex type
        vertex_activation = vertex_probs.mean()
        edge_activation = edge_probs.mean()
        triangle_activation = triangle_probs.mean()
        tetra_activation = tetra_probs.mean()
        
        # Compute entropy across simplex types to encourage diversity
        probs = torch.stack([
            vertex_activation, edge_activation,
            triangle_activation, tetra_activation
        ])
        # Normalize to create a proper probability distribution
        probs = probs / (probs.sum() + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        # Negate entropy since we want to minimize the loss
        entropy_loss = -0.1 * entropy

        torch.stack([vertex_probs, edge_probs, triangle_probs, tetra_probs])

        return entropy_loss

    def get_active_simplex_embeddings(self, vertices, edges, triangles, tetra, device):
        """Get embeddings only for active simplices."""
        # Get indices of active simplices
        active_vertices = vertices.nonzero().squeeze(-1)
        active_edges =  edges.nonzero().squeeze(-1)
        active_triangles =  triangles.nonzero().squeeze(-1)
        active_tetra =  tetra.nonzero().squeeze(-1)

        print("\nActive simplex counts:")
        print(f"Vertices: {len(active_vertices)}")
        print(f"Edges: {len(active_edges)}")
        print(f"Triangles: {len(active_triangles)}")
        print(f"Tetrahedra: {len(active_tetra)}")

        # Get embeddings only for active simplices
        def get_embedding(embedding_layer, active_indices, probs):
            embeddings = embedding_layer(active_indices)
            probs_expanded = probs[active_indices].unsqueeze(-1)
            result = embeddings * probs_expanded
            # print(f"Embedding shape: {result.shape}")
            return result

        embeddings = {
            'rank_0': get_embedding(self.vertex_embeddings, active_vertices, vertices),
            'rank_1': get_embedding(self.edge_embeddings, active_edges, edges),
            'rank_2': get_embedding(self.triangle_embeddings, active_triangles, triangles),
            'rank_3': get_embedding(self.tetra_embeddings, active_tetra, tetra)
        }

        embeddings['active_indices'] = {
            'vertices': active_vertices,
            'edges': active_edges,
            'triangles': active_triangles,
            'tetra': active_tetra
        }

        return embeddings

    def build_sparse_complex_matrices(self, rectified):
        """Build sparse matrices only for active simplices."""
        active_vertices = rectified.vertices.nonzero().squeeze(-1)
        
        # If no vertices are active, return None
        if len(active_vertices) == 0:
            return None

        # Get the subset of constraint matrices for active vertices
        active_edges = rectified.edges.nonzero().squeeze(-1)
        active_triangles = rectified.triangles.nonzero().squeeze(-1)
        active_tetra = rectified.tetra.nonzero().squeeze(-1)

        # Create sparse matrices only for active simplices
        matrices = {
            'vertices': rectified.vertices,
            'edges': rectified.edges,
            'triangles': rectified.triangles,
            'tetra': rectified.tetra,
            'boundary_1': self.constraints.boundary_1[active_edges][:, active_vertices] if len(active_edges) > 0 else None,
            'boundary_2': self.constraints.boundary_2[active_triangles][:, active_edges] if len(active_triangles) > 0 else None,
            'boundary_3': self.constraints.boundary_3[active_tetra][:, active_triangles] if len(active_tetra) > 0 else None
        }

        return matrices
    
    def split_simplices(self, logits):
        vertices = logits[:self.num_vertices] + F.relu(self.vertex_bias)
        edges = logits[self.num_vertices:self.num_vertices + self.num_edges]
        triangles = logits[self.num_vertices + self.num_edges:self.num_vertices + self.num_edges + self.num_triangles]
        tetrahedra = logits[-self.num_tetra:]

        return vertices, edges, triangles, tetrahedra

    def compute_contrastive_loss(self, logits, function='InfoNCE', temperature=0.1):
        if function == 'InfoNCE':
            # Normalize embeddings
            anchor = F.normalize(logits[0], dim=1)
            positive = F.normalize(logits[1], dim=1)
            negatives = F.normalize(logits[2:], dim=1)
            
            # Compute logits
            logits_pos = torch.einsum('nc,nc->n', [anchor, positive])
            logits_neg = torch.einsum('nc,nkc->nk', [anchor, negatives])
            
            # Concatenate logits and compute loss
            logits = torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
            
            return F.cross_entropy(logits / temperature, labels)
        
        elif function == 'Triplet':
            anchor = logits[0]
            positive = logits[1]
            negative = logits[2]
            return F.triplet_margin_with_distance_loss(anchor, positive, negative)
        else:
            raise ValueError(f"{function} is not a valid contrastive loss function")

    def generate_complex(self, logits):
        logits = logits[:self.num_vertices] + F.relu(self.vertex_bias)
        if not self.hard:
            simplex_probs = self.gumbel(logits)
            vertex_loss = 0

        else:
            simplex_probs = F.sigmoid(logits / self.temp)
        
        vertex_probs, edge_probs, triangle_probs, tetra_probs = self.split_simplices(simplex_probs)

        rectified = enforce_constraints(
            vertex_probs, edge_probs, triangle_probs, tetra_probs, self.constraints
        )

        # entropy_loss = self.compute_entropy_loss(rectified.vertices, rectified.edges, rectified.triangles, rectified.tetra)

        if self.hard:
            vertices = torch.bernoulli(rectified.vertices)
            edges = torch.bernoulli(rectified.edges)
            triangles = torch.bernoulli(rectified.triangles)
            tetrahedra = torch.bernoulli(rectified.tetra)

            rectified2 = enforce_constraints(
                vertices, edges, triangles, tetrahedra, self.constraints
            )

            # vertex_loss = self.compute_vertex_penalty(rectified.vertices)
            vertex_logits, edge_logits, triangle_logits, tetra_logits = self.split_simplices(logits)

            vertices = vertex_logits + (rectified2.vertices - vertex_logits).detach()
            edges = edge_logits + (rectified2.edges - edge_logits).detach()
            triangles = triangle_logits + (rectified2.triangles - triangle_logits).detach()
            tetrahedra = tetra_logits + (rectified2.tetra - tetra_logits).detach()

        else:
            vertices = rectified.vertices
            edges = rectified.edges
            triangles = rectified.triangles
            tetrahedra = rectified.tetra

        if torch.sum(vertices) == 0:
            return None, None, None
        
        # diversity_loss = {
        #     'binary_entropy': entropy_loss,
        #     'diversity': vertex_loss
        # }

        active_embeddings = self.get_active_simplex_embeddings(vertices, edges, triangles, tetrahedra, logits.device)

        self.active_simplices = active_embeddings['active_indices']

        embeddings = {
            'rank_0': active_embeddings['rank_0'],
            'rank_1': active_embeddings['rank_1'],
            'rank_2': active_embeddings['rank_2'],
            'rank_3': active_embeddings['rank_3']
        }

        complex_matrices = build_sparse_matrices(rectified, self.constraints, active_embeddings['active_indices'])

        embeddings = self.get_active_simplex_embeddings(vertices, edges, triangles, tetrahedra, logits.device)

        return embeddings, complex_matrices

    def forward(self, x):
        # print("\nEncoder Forward Pass Shapes:")
        # print(f"Input shape: {x.shape}")  # Should be [batch, bands, time]
        
        # Process each frequency band independently
        band_features = []
        for i, band_processor in enumerate(self.band_processors):
            band = x[:, i:i+1]
            # print(f"Single band {i} shape: {band.shape}")
            processed = band_processor(band)
            # print(f"Processed band {i} shape: {processed.shape}")
            band_features.append(processed)
        
        # Concatenate band features
        x = torch.cat(band_features, dim=1)
        # print(f"After concatenating all bands: {x.shape}")
        
        # Create skip connection with matched dimensions
        skip = self.skip_maxpool(x.transpose(1, 2)).transpose(1, 2)
        # print(f"Skip connection shape: {skip.shape}")
        
        # Cross-band processing
        x = self.cross_band(x)
        # print(f"After cross-band processing: {x.shape}")
        
        # Add skip connection with learnable weight
        x = x + self.skip_weight * skip
        # print(f"After adding skip connection: {x.shape}")
        
        # Temporal reduction
        x = self.temporal_reduction(x)
        # print(f"After temporal reduction: {x.shape}")
        
        # Flatten and process through MLP
        x = x.flatten(1)
        # print(f"After flattening: {x.shape}")
        logits = self.to_simplices(x).squeeze()

        # Add contrastive loss while training
        contrastive_loss = None
        if self.training:
            contrastive_loss = self.compute_contrastive_loss(logits), contrastive_loss

        return self.generate_complex(logits), contrastive_loss

class ScaleLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return x * self.scale_factor



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
    ...