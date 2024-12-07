import torch
import itertools
from typing import Dict, List, Tuple
from dataclasses import dataclass
torch.autograd.set_detect_anomaly(True)

@dataclass
class SimplexIndices:
    edges: torch.Tensor  # shape (10, 2) for 5 vertices
    triangles: torch.Tensor  # shape (10, 3) for 5 vertices
    tetra: torch.Tensor # shape (5, 4) for 5 vertices

class ConstraintMatrices:
    def __init__(self, 
                 v2e: torch.Tensor, 
                 e2t: torch.Tensor, 
                 t2tt: torch.Tensor,
                 indices: SimplexIndices):
        self.vertex_to_edge = v2e
        self.edge_to_triangle = e2t
        self.triangle_to_tetra = t2tt
        self.indices = indices

    @classmethod
    def create(cls, n_vertices: int) -> 'ConstraintMatrices':
        # Create index mappings as tensors
        vertex_indices = list(range(n_vertices))
        edge_indices = torch.tensor(list(itertools.combinations(vertex_indices, 2)), requires_grad=False)  # shape (10, 2)
        triangle_indices = torch.tensor(list(itertools.combinations(vertex_indices, 3)), requires_grad=False)  # shape (10, 3)
        tetra_indices = torch.tensor(list(itertools.combinations(vertex_indices, 4)), requires_grad=False)  # shape (5, 4)

        # Create vertex-to-edge constraint matrix
        v2e = torch.zeros(len(edge_indices), n_vertices)
        for i, (v1, v2) in enumerate(edge_indices):
            v2e[i, v1] = 1
            v2e[i, v2] = 1

        # Create edge-to-triangle constraint matrix
        e2t = torch.zeros(len(triangle_indices), len(edge_indices))
        for i, (v1, v2, v3) in enumerate(triangle_indices):
            # Find indices of constituent edges
            e1 = (edge_indices == torch.tensor([v1, v2])).all(dim=1).nonzero().item()
            e2 = (edge_indices == torch.tensor([v1, v3])).all(dim=1).nonzero().item()
            e3 = (edge_indices == torch.tensor([v2, v3])).all(dim=1).nonzero().item()
            e2t[i, [e1, e2, e3]] = 1

        # Create triangle-to-tetra constraint matrix
        t2tt = torch.zeros(len(tetra_indices), len(triangle_indices))
        for i, (v1, v2, v3, v4) in enumerate(tetra_indices):
            # Find indices of constituent triangles
            t1 = (triangle_indices == torch.tensor([v1, v2, v3])).all(dim=1).nonzero().item()
            t2 = (triangle_indices == torch.tensor([v1, v2, v4])).all(dim=1).nonzero().item()
            t3 = (triangle_indices == torch.tensor([v1, v3, v4])).all(dim=1).nonzero().item()
            t4 = (triangle_indices == torch.tensor([v2, v3, v4])).all(dim=1).nonzero().item()
            t2tt[i, [t1, t2, t3, t4]] = 1

        # Store index tensors in SimplexIndices
        indices = SimplexIndices(
            edges=edge_indices,
            triangles=triangle_indices,
            tetra=tetra_indices
        )

        return cls(v2e, e2t, t2tt, indices)


@dataclass
class RectifiedProbs:
    vertices: torch.Tensor
    edges: torch.Tensor
    triangles: torch.Tensor
    tetra: torch.Tensor
    all_simplices: torch.Tensor

def enforce_constraints(
    vertex_probs: torch.Tensor,
    edge_probs: torch.Tensor, 
    triangle_probs: torch.Tensor,
    tetra_probs: torch.Tensor,
    matrices: ConstraintMatrices,
    eps: float = 1e-10
) -> RectifiedProbs:
    """
    Enforce simplicial complex constraints through vectorized operations.
    Returns rectified probabilities that satisfy the constraints.
    """
    # Compute vertex-constrained edge probabilities
    vertex_pairs = vertex_probs[matrices.indices.edges]
    zero_mask = (vertex_pairs == 0).any(dim=1)
    vertex_pairs_log_mean = torch.exp(
        torch.log(vertex_pairs + eps).sum(dim=1) / 2
    )
    # Zero out probabilities while maintaining gradients
    vertex_pairs_log_mean = torch.where(zero_mask, 
                                      vertex_pairs_log_mean - vertex_pairs_log_mean,
                                      vertex_pairs_log_mean)
    rectified_edge_probs = torch.minimum(edge_probs, vertex_pairs_log_mean)
    
    # Rectify triangle probabilities using edge probabilities
    edge_log_probs = torch.log(rectified_edge_probs + eps)
    edge_triples = torch.matmul(matrices.edge_to_triangle, edge_log_probs)
    edge_triplets_log_mean = torch.exp(edge_triples / 3)
    zero_edges = (rectified_edge_probs == 0)
    zero_triangles = (matrices.edge_to_triangle @ zero_edges.float()).bool()
    edge_triplets_log_mean = torch.where(zero_triangles.squeeze(),
                                       edge_triplets_log_mean - edge_triplets_log_mean,
                                       edge_triplets_log_mean)
    rectified_triangle_probs = torch.minimum(triangle_probs, edge_triplets_log_mean)

    # Rectify tetrahedron probabilities 
    triangle_log_probs = torch.log(rectified_triangle_probs + eps)
    triangle_quads = torch.matmul(matrices.triangle_to_tetra, triangle_log_probs)
    triangle_quads_log_mean = torch.exp(triangle_quads / 4)
    zero_triangles = (rectified_triangle_probs == 0)
    zero_tetra = (matrices.triangle_to_tetra @ zero_triangles.float()).bool()
    triangle_quads_log_mean = torch.where(zero_tetra.squeeze(),
                                        triangle_quads_log_mean - triangle_quads_log_mean,
                                        triangle_quads_log_mean)
    rectified_tetra_probs = torch.minimum(tetra_probs, triangle_quads_log_mean)

    return RectifiedProbs(
        vertices=vertex_probs,
        edges=rectified_edge_probs,
        triangles=rectified_triangle_probs,
        tetra=rectified_tetra_probs,
        all_simplices=torch.cat([vertex_probs, rectified_edge_probs, rectified_triangle_probs, rectified_tetra_probs])
    )

def verify_constraints(probs: RectifiedProbs, matrices: ConstraintMatrices) -> None:
    """Verify that the rectified probabilities satisfy the constraints."""
    print("\nVerifying constraints:")
    print(f"Vertices: {probs.vertices}")
    print(f"Edges: {probs.edges}")
    # print(f"Triangles: {probs.triangles}")
    # print(f"Tetrahedra: {probs.tetra}")
    
    # Verify edge constraints
    print("\nEdge Constraints:")
    for i, (v1, v2) in enumerate(matrices.indices.edges):
        vertex_constraint = torch.sqrt(probs.vertices[v1] * probs.vertices[v2])
        print(f"Edge {v1}-{v2}:")
        print(f"  Rectified prob: {probs.edges[i]:.3f}")
        print(f"  Vertex constraint: {vertex_constraint:.3f}")
        print(f"  Original prob: {(probs.edges[i]**2 / vertex_constraint):.3f}")
    
    # Verify triangle constraints
    print("\nTriangle Constraints:")
    for i, (v1, v2, v3) in enumerate(matrices.indices.triangles):
        # Get the indices of the three edges that form this triangle
        e1 = (matrices.indices.edges == torch.tensor([v1, v2])).all(dim=1).nonzero().item()
        e2 = (matrices.indices.edges == torch.tensor([v1, v3])).all(dim=1).nonzero().item()
        e3 = (matrices.indices.edges == torch.tensor([v2, v3])).all(dim=1).nonzero().item()
        
        # Compute geometric mean of edge probabilities
        edge_constraint = torch.pow(
            probs.edges[e1] * probs.edges[e2] * probs.edges[e3],
            1/3
        )
        
        if v1 == 1:
            print(f"Triangle {v1}-{v2}-{v3}:")
            print(f"  Rectified prob: {probs.triangles[i]:.3f}")
            print(f"  Edge constraint: {edge_constraint:.3f}")
            print(f"  Original prob: {(probs.triangles[i]**2 / edge_constraint):.3f}")
            print(f"  Contributing edges: {probs.edges[e1]:.3f}, {probs.edges[e2]:.3f}, {probs.edges[e3]:.3f}")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    n_vertices = 7
    n_edges = len(torch.combinations(torch.tensor(range(n_vertices)), 2))
    print(n_edges)
    n_triangles = len(torch.combinations(torch.tensor(range(n_vertices)), 3))
    n_tetrahedra = len(torch.combinations(torch.tensor(range(n_vertices)), 4))

    
    # Create random probabilities
    vertex_probs = torch.rand(n_vertices)
    vertex_probs[1] = 0
    vertex_probs[5] = 0
    edge_probs = torch.rand(n_edges) 
    edge_probs[10] = 0
    # edge_probs[5] = 0
    triangle_probs = torch.rand(n_triangles) 
    tetra_probs = torch.rand(n_tetrahedra) 
    
    # Create constraint matrices
    matrices = ConstraintMatrices.create(n_vertices)
    
    # Enforce constraints
    rectified = enforce_constraints(
        vertex_probs, edge_probs, triangle_probs, tetra_probs, matrices
    )
    
    # Verify constraints
    verify_constraints(rectified, matrices)

if __name__ == "__main__":
    main()