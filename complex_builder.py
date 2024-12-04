from dataclasses import dataclass
import torch
from rectifier import RectifiedProbs
from rectifier import ConstraintMatrices
torch.autograd.set_detect_anomaly(True)


@dataclass
class SparseSimplicialMatrices:
    # Adjacency matrices
    adjacencies: dict[str, torch.Tensor]
    
    # Incidence matrices
    incidences: dict[str, torch.Tensor]

def dense_to_sparse(dense_tensor: torch.Tensor) -> torch.Tensor:
    """Convert dense tensor to sparse COO tensor"""
    indices = torch.nonzero(dense_tensor).t()
    values = dense_tensor[indices[0], indices[1]]
    return torch.sparse_coo_tensor(indices, values, dense_tensor.size()).coalesce()

def build_sparse_matrices(probs: RectifiedProbs, matrices: ConstraintMatrices) -> SparseSimplicialMatrices:
    """Build sparse matrices by first constructing dense then converting to sparse"""
    n_vertices = len(probs.vertices)
    
    # Build vertex adjacency matrix - ensure we maintain gradients
    vertex_adjacency = torch.zeros((n_vertices, n_vertices), 
                                 device=probs.edges.device, 
                                 dtype=probs.edges.dtype)
    edge_indices = matrices.indices.edges
    vertex_adjacency[edge_indices[:, 0], edge_indices[:, 1]] = probs.edges
    vertex_adjacency[edge_indices[:, 1], edge_indices[:, 0]] = probs.edges  # Symmetric
    
    # Build incidence matrices
    vertex_edge_incidence = matrices.vertex_to_edge.T * probs.edges.unsqueeze(0)
    edge_triangle_incidence = matrices.edge_to_triangle.T * probs.triangles.unsqueeze(0)
    triangle_tetra_incidence = matrices.triangle_to_tetra.T * probs.tetra.unsqueeze(0)
    
    # Build higher-order adjacencies with gradient-preserving operations
    edge_adjacency = edge_triangle_incidence @ edge_triangle_incidence.t()
    eye = torch.eye(edge_adjacency.shape[0], device=edge_adjacency.device)
    edge_adjacency = edge_adjacency - eye * edge_adjacency
    
    triangle_adjacency = triangle_tetra_incidence @ triangle_tetra_incidence.t()
    eye = torch.eye(triangle_adjacency.shape[0], device=triangle_adjacency.device)
    triangle_adjacency = triangle_adjacency - eye * triangle_adjacency
    
    tetrahedra_adjacency = triangle_tetra_incidence.t() @ triangle_tetra_incidence
    eye = torch.eye(tetrahedra_adjacency.shape[0], device=tetrahedra_adjacency.device)
    tetrahedra_adjacency = tetrahedra_adjacency - eye * tetrahedra_adjacency
    
    # Modified sparse conversion to ensure gradient flow
    def to_sparse_with_gradients(dense_tensor):
        indices = torch.nonzero(dense_tensor).t()
        values = dense_tensor[indices[0], indices[1]]
        sparse = torch.sparse_coo_tensor(indices, values, dense_tensor.size())
        # Force coalescing to ensure consistent gradient flow
        return sparse.coalesce()
    
    def register_grad_hook(tensor, name, scale_factor=10.0):
        if tensor.requires_grad:
            def hook(grad):
                scaled_grad = grad * scale_factor
                # print(f"{name} grad norm before scaling: {grad.norm().item()}")
                # print(f"{name} grad norm after scaling: {scaled_grad.norm().item()}")
                return scaled_grad
            tensor.register_hook(hook)

    complex_matrices = SparseSimplicialMatrices(
        adjacencies={
            'rank_0': to_sparse_with_gradients(vertex_adjacency),
            'rank_1': to_sparse_with_gradients(edge_adjacency),
            'rank_2': to_sparse_with_gradients(triangle_adjacency),
            'rank_3': to_sparse_with_gradients(tetrahedra_adjacency)
        },
        incidences={
            'rank_1': to_sparse_with_gradients(vertex_edge_incidence),
            'rank_2': to_sparse_with_gradients(edge_triangle_incidence),
            'rank_3': to_sparse_with_gradients(triangle_tetra_incidence)
        }
    )

    for rank, matrix in complex_matrices.adjacencies.items():
        register_grad_hook(matrix, f"Adjacency matrix {rank}")
    for rank, matrix in complex_matrices.incidences.items():
        register_grad_hook(matrix, f"Incidence matrix {rank}")

    return complex_matrices

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