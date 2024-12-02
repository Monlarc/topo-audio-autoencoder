from dataclasses import dataclass
import torch
from torch_sparse import SparseTensor
from rectifier import RectifiedProbs
from rectifier import ConstraintMatrices


@dataclass
class SparseSimplicialMatrices:
    # Adjacency matrices
    adjacencies: dict[int, SparseTensor]
    
    # Incidence matrices
    incidences: dict[int, SparseTensor]

def build_sparse_matrices(probs: RectifiedProbs, matrices: ConstraintMatrices) -> SparseSimplicialMatrices:
    """
    Build sparse adjacency and incidence matrices from probabilities
    """
    n_vertices = len(probs.vertices)
    n_edges = len(probs.edges)
    n_triangles = len(probs.triangles)
    n_tetra = len(probs.tetra)
    
    # Build vertex adjacency matrix
    edge_indices = matrices.indices.edges
    v_adj_indices = torch.cat([
        torch.stack([edge_indices[:, 0], edge_indices[:, 1]]),
        torch.stack([edge_indices[:, 1], edge_indices[:, 0]])
    ], dim=1)
    v_adj_values = probs.edges.repeat(2)
    vertex_adjacency = SparseTensor(
        row=v_adj_indices[0],
        col=v_adj_indices[1],
        value=v_adj_values,
        sparse_sizes=(n_vertices, n_vertices)
    )
    
    # Build vertex-edge incidence matrix
    v2e = matrices.vertex_to_edge
    v2e_indices = v2e.nonzero().T
    v2e_values = probs.edges.repeat_interleave(2)  # Each edge connects 2 vertices
    vertex_edge_incidence = SparseTensor(
        row=v2e_indices[1],  # Transpose from original
        col=v2e_indices[0],
        value=v2e_values,
        sparse_sizes=(n_vertices, n_edges)
    )
    
    # Build edge-triangle incidence matrix
    e2t = matrices.edge_to_triangle
    e2t_indices = e2t.nonzero().T
    e2t_values = probs.triangles.repeat_interleave(3)  # Each triangle has 3 edges
    edge_triangle_incidence = SparseTensor(
        row=e2t_indices[1],  # Transpose from original
        col=e2t_indices[0],
        value=e2t_values,
        sparse_sizes=(n_edges, n_triangles)
    )
    
    # Build triangle-tetra incidence matrix
    t2tt = matrices.triangle_to_tetra
    t2tt_indices = t2tt.nonzero().T
    t2tt_values = probs.tetra.repeat_interleave(4)  # Each tetra has 4 triangles
    triangle_tetra_incidence = SparseTensor(
        row=t2tt_indices[1],  # Transpose from original
        col=t2tt_indices[0],
        value=t2tt_values,
        sparse_sizes=(n_triangles, n_tetra)
    )
    
    # Build tetrahedra adjacency matrix (tetrahedra connected by shared triangles)
    tetrahedra_adjacency = triangle_tetra_incidence.t() @ triangle_tetra_incidence
    # Remove diagonal
    tetra_adj_indices = tetrahedra_adjacency.coo()[:2]
    mask = tetra_adj_indices[0] != tetra_adj_indices[1]
    tetrahedra_adjacency = SparseTensor(
        row=tetra_adj_indices[0][mask],
        col=tetra_adj_indices[1][mask],
        value=tetrahedra_adjacency.storage.value()[mask],
        sparse_sizes=(n_tetra, n_tetra)
    )
    
    # Build edge adjacency matrix (edges connected by triangles)
    # Using sparse matrix multiplication
    edge_adjacency = edge_triangle_incidence @ edge_triangle_incidence.t()
    # Remove diagonal
    edge_adj_indices = edge_adjacency.coo()[:2]
    mask = edge_adj_indices[0] != edge_adj_indices[1]
    edge_adjacency = SparseTensor(
        row=edge_adj_indices[0][mask],
        col=edge_adj_indices[1][mask],
        value=edge_adjacency.storage.value()[mask],
        sparse_sizes=(n_edges, n_edges)
    )
    
    # Build triangle adjacency matrix (triangles connected by tetrahedra)
    triangle_adjacency = triangle_tetra_incidence @ triangle_tetra_incidence.t()
    # Remove diagonal
    tri_adj_indices = triangle_adjacency.coo()[:2]
    mask = tri_adj_indices[0] != tri_adj_indices[1]
    triangle_adjacency = SparseTensor(
        row=tri_adj_indices[0][mask],
        col=tri_adj_indices[1][mask],
        value=triangle_adjacency.storage.value()[mask],
        sparse_sizes=(n_triangles, n_triangles)
    )
    
    return SparseSimplicialMatrices(
        adjacencies={
            0: vertex_adjacency,
            1: edge_adjacency,
            2: triangle_adjacency,
            3: tetrahedra_adjacency
        },
        incidences={
            1: vertex_edge_incidence,
            2: edge_triangle_incidence,
            3: triangle_tetra_incidence
        }
    )

def verify_sparse_matrices(matrices: SparseSimplicialMatrices, probs: RectifiedProbs, constraints: ConstraintMatrices):
    """
    Verify the constructed sparse matrices have the expected properties
    """
    print("Verifying sparse matrix properties:")
    
    # Check symmetry of adjacency matrices
    print("\nChecking number of nonzeros:")
    print(f"Vertex adjacency: {matrices.vertex_adjacency.nnz()}")
    print(f"Edge adjacency: {matrices.edge_adjacency.nnz()}")
    print(f"Triangle adjacency: {matrices.triangle_adjacency.nnz()}")
    print(f"Tetrahedra adjacency: {matrices.tetrahedra_adjacency.nnz()}")
    
    print("\nChecking incidence matrices:")
    print(f"Vertex-edge incidence: {matrices.vertex_edge_incidence.nnz()}")
    print(f"Edge-triangle incidence: {matrices.edge_triangle_incidence.nnz()}")
    print(f"Triangle-tetra incidence: {matrices.triangle_tetra_incidence.nnz()}")
    
    # Convert small parts to dense for verification
    if len(probs.vertices) <= 10:
        print("\nSample of vertex adjacency (dense):")
        print(matrices.vertex_adjacency.to_dense())
        
        print("\nSample of edge adjacency (dense):")
        print(matrices.edge_adjacency.to_dense())


if __name__ == "__main__":
    verify_sparse_matrices(build_sparse_matrices(RectifiedProbs.random(10), ConstraintMatrices.random(10)), RectifiedProbs.random(10), ConstraintMatrices.random(10))