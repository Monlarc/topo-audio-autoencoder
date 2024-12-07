import torch
from custom_sccn import GradientSCCN

def test_sccn_gradients_realistic():
    # Create larger features and matrices
    n_vertices = 20
    n_edges = n_vertices * 2  # Approximate
    features = {
        'rank_0': torch.randn(n_vertices, 64, requires_grad=True),
        'rank_1': torch.randn(n_edges, 64, requires_grad=True)
    }
    
    # Create sparser adjacency matrices
    adj_0_indices = torch.randint(0, n_vertices, (2, 5))  # Only 5 connections
    adj_0 = torch.sparse_coo_tensor(
        indices=adj_0_indices,
        values=torch.ones(5),
        size=(n_vertices, n_vertices)
    ).requires_grad_()
    
    adj_1_indices = torch.randint(0, n_edges, (2, 5))
    adj_1 = torch.sparse_coo_tensor(
        indices=adj_1_indices,
        values=torch.ones(5),
        size=(n_edges, n_edges)
    ).requires_grad_()
    
    # Create sparse incidence matrix
    inc_indices = torch.randint(0, min(n_vertices, n_edges), (2, 5))
    inc = torch.sparse_coo_tensor(
        indices=inc_indices,
        values=torch.ones(5),
        size=(n_vertices, n_edges)
    ).requires_grad_()
    
    # Create multi-layer SCCN
    sccn = GradientSCCN(channels=64, max_rank=1, n_layers=4)
    
    # Forward pass
    out = sccn(
        features, 
        {'rank_1': inc}, 
        {'rank_0': adj_0, 'rank_1': adj_1}
    )
    
    # Compute loss and backward
    loss = sum(x.sum() for x in out.values())
    print(f"\nLoss value: {loss.item()}")
    loss.backward()
    
    # Print gradients for input features and matrices
    print("\nInput gradients:")
    print(f"Features rank 0 grad norm: {features['rank_0'].grad.norm().item()}")
    print(f"Features rank 1 grad norm: {features['rank_1'].grad.norm().item()}")
    print(f"Adjacency 0 grad norm: {adj_0.grad.norm().item() if adj_0.grad is not None else 'None'}")
    print(f"Adjacency 1 grad norm: {adj_1.grad.norm().item() if adj_1.grad is not None else 'None'}")
    print(f"Incidence grad norm: {inc.grad.norm().item() if inc.grad is not None else 'None'}")
    
    # Print gradients for each layer
    print("\nLayer gradients:")
    for name, param in sccn.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item()}")
        else:
            print(f"{name} has no gradient")

if __name__ == "__main__":
    test_sccn_gradients_realistic()