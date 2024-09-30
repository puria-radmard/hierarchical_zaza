import torch
from torch_geometric.transforms import BaseTransform


class NodeReorder(BaseTransform):
    """
    Sorts the nodes based on their positions along the y = x line;
    reorders based on projections onto the [1, 1] vector.
    Add positional embeddings and adjacency matrix.
    """
    def __init__(self, unit_vec=None):
        if unit_vec is None:
            self.unit_vec = torch.tensor([1., 1.]) / (2 ** 0.5)  # univec along y = x
        else:
            self.unit_vec = torch.tensor(unit_vec)
        
    def forward(self, data):
        # Compute projections onto the unit vector
        projections = data.pos @ self.unit_vec
        # Get new order by sorting the projections
        new_order = projections.argsort()
        # Create mapping
        mapping = torch.zeros_like(new_order)
        mapping[new_order] = torch.arange(len(new_order), dtype=torch.long)

        # Reorder node features
        if hasattr(data, 'laplacian_eigen_pe'):
            data.x = torch.cat([data.x, data.laplacian_eigen_pe], dim=1)
            del data.laplacian_eigen_pe

        data.x = data.x[new_order]
        data.pos = data.pos[new_order]

        # Update edge indices
        data.edge_index = mapping[data.edge_index]
        
        # Compute adjacency matrix A directly using PyTorch tensors
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        A[edge_index[0], edge_index[1]] = 1.0
        data.adj = A  # Add adjacency matrix to data

        return data