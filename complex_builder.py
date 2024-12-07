from dataclasses import dataclass
import torch
from rectifier import RectifiedProbs
from rectifier import ConstraintMatrices
from typing import Dict
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

def build_sparse_matrices(probs: RectifiedProbs, matrices: ConstraintMatrices, active_indices: Dict[str, torch.Tensor]) -> SparseSimplicialMatrices:
    """Build sparse matrices by selecting active rows and columns."""
    # Debug prints
    # print("\nActive indices:")
    # for rank, indices in active_indices.items():
    #     print(f"{rank}: {indices.shape}")
    
    if len(active_indices['vertices']) == 0:
        print("No active vertices!")
        return None
        
    # Build vertex adjacency matrix
    vertex_adjacency = torch.zeros((len(probs.vertices), len(probs.vertices)), 
                                 device=probs.edges.device, 
                                 dtype=probs.edges.dtype)
    edge_indices = matrices.indices.edges
    vertex_adjacency[edge_indices[:, 0], edge_indices[:, 1]] = probs.edges
    vertex_adjacency[edge_indices[:, 1], edge_indices[:, 0]] = probs.edges
    
    # Debug prints before and after selection
    # print("\nMatrix shapes before selection:")
    # print(f"vertex_adjacency: {vertex_adjacency.shape}")
    
    # Select active submatrices
    vertex_adjacency = vertex_adjacency[active_indices['vertices']][:, active_indices['vertices']]
    

    
    # Build incidence matrices
    vertex_edge_incidence = matrices.vertex_to_edge.T * probs.edges.unsqueeze(0)
    edge_triangle_incidence = matrices.edge_to_triangle.T * probs.triangles.unsqueeze(0)
    triangle_tetra_incidence = matrices.triangle_to_tetra.T * probs.tetra.unsqueeze(0)
    
    # Select active rows and columns for incidence matrices
    vertex_edge_incidence = vertex_edge_incidence[active_indices['vertices']][:, active_indices['edges']]
    edge_triangle_incidence = edge_triangle_incidence[active_indices['edges']][:, active_indices['triangles']]
    triangle_tetra_incidence = triangle_tetra_incidence[active_indices['triangles']][:, active_indices['tetra']]
    
    # Build higher-order adjacencies
    edge_adjacency = edge_triangle_incidence @ edge_triangle_incidence.t()
    triangle_adjacency = triangle_tetra_incidence @ triangle_tetra_incidence.t()
    tetra_adjacency = triangle_tetra_incidence.t() @ triangle_tetra_incidence
    
    # Remove self-loops without using in-place operations
    eye = torch.eye
    edge_adjacency = edge_adjacency * (1 - eye(len(active_indices['edges']), device=edge_adjacency.device))
    triangle_adjacency = triangle_adjacency * (1 - eye(len(active_indices['triangles']), device=triangle_adjacency.device))
    tetra_adjacency = tetra_adjacency * (1 - eye(len(active_indices['tetra']), device=tetra_adjacency.device))

    # print("\nMatrix shapes after selection:")
    # print(f"vertex_adjacency: {vertex_adjacency.shape}")
    # print(f"edge_adjacency: {edge_adjacency.shape}")
    # print(f"triangle_adjacency: {triangle_adjacency.shape}")
    # print(f"tetra_adjacency: {tetra_adjacency.shape}")

    # print("vertex_edge_incidence:", vertex_edge_incidence.shape)
    # print("edge_triangle_incidence:", edge_triangle_incidence.shape)
    # print("triangle_tetra_incidence:", triangle_tetra_incidence.shape)
    
    def to_sparse(dense):
        indices = torch.nonzero(dense).t()
        values = dense[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, dense.size()).coalesce()
    
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
            'rank_0': to_sparse(vertex_adjacency),
            'rank_1': to_sparse(edge_adjacency),
            'rank_2': to_sparse(triangle_adjacency),
            'rank_3': to_sparse(tetra_adjacency)
        },
        incidences={
            'rank_1': to_sparse(vertex_edge_incidence),
            'rank_2': to_sparse(edge_triangle_incidence),
            'rank_3': to_sparse(triangle_tetra_incidence)
        }
    )

    # for rank, matrix in complex_matrices.adjacencies.items():
    #     register_grad_hook(matrix, f"Adjacency matrix {rank}")
    # for rank, matrix in complex_matrices.incidences.items():
    #     register_grad_hook(matrix, f"Incidence matrix {rank}")

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