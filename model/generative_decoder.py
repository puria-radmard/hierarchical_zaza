import torch
from torch import nn


class generative_model(nn.Module):
    def __init__(self, hparams: dict):
        super(generative_model, self).__init__()
        self.hparams = hparams

        zi_dim = hparams['zi_dim']                      # F
        node_feature_dim = hparams['node_feature_dim']  # D
        attention_dim = hparams['attention_dim']        # Dimension for Q, K, V 
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query_linear = nn.Linear(zi_dim, attention_dim)
        self.key_linear = nn.Linear(zi_dim, attention_dim)
        self.value_linear = nn.Linear(zi_dim, attention_dim)

        # Node feature decoder (reconstructs X from z_i and A_hat)
        # GCN and MLP?
        # self.node_decoder = 
        
    def adj_decoder(self, z_i, batch):
        """
        Reconstruct adjacency matrices for each graph.

        Args:
            z_i (Tensor): Node-level latent variables [num_nodes, zi_dim].
            batch (Tensor): Batch vector assigning each node to a graph in the batch.

        Returns:
            A_hat_batch (Tensor): Reconstructed adjacency matrices [batch, num_nodes, num_nodes].
        """
        num_graphs = batch.max().item() + 1
        A_hat_list = []

        for i in range(num_graphs):
            mask = (batch == i)
            Z = z_i[mask]                   # [num_nodes_in_graph_i, zi_dim]

            # Compute adjacency matrix reconstruction for each graph
            
            # ===== Attention decoder =====
            # Compute Query, Key, Value
            Q = self.query_linear(Z)        # [num_nodes_in_graph_i, attention_dim]
            K = self.key_linear(Z)          # [num_nodes_in_graph_i, attention_dim]
            V = self.value_linear(Z)        # [num_nodes_in_graph_i, attention_dim]
            
            attention_weights = Q @ K.T                        # [num_nodes_in_graph_i, num_nodes_in_graph_i]
            attention_weights = attention_weights.softmax(1)   # same shape, but attention_weights.sum(1) = ones
            weighted_values = (attention_weights.unsqueeze(-1) * V.unsqueeze(0)).sum(1)   # [num_nodes_in_graph_i, attention_dim]
            Z_att = weighted_values / (self.attention_dim**0.5)
        
            A_hat = torch.sigmoid(torch.matmul(Z_att, Z_att.t()))  # [num_nodes_in_graph_i, num_nodes_in_graph_i]
            
            # Diag(A_hat) is ~ 1, while Diag(A) is 0
            # ***Options - 1) set diag(A_hat) to 0; 2) set diag(A) to 1 for BCE
            A_hat.fill_diagonal_(0) # Option 1
            A_hat_list.append(A_hat)

        return torch.stack(A_hat_list, dim=0)        

    def forward(self, z_i, batch):
        """
        Forward pass through the decoder.

        Args:
            z_i (Tensor): Node-level latent variables [num_nodes, zi_dim].
            batch (Tensor): Batch vector assigning each node to a graph in the batch.

        Returns:
            X_hat (Tensor): Reconstructed node features.
        """
        device = z_i.device
        num_nodes = z_i.size(0)
        num_graphs = batch.max().item() + 1
        
        # Reconstruct A
        A_hat_batch = self.adj_decoder(z_i, batch)

        # Initialize reconstructed node features
        X_hat = torch.zeros((num_nodes, self.hparams['node_feature_dim']), device=device)
        
#         # Process each graph individually
#         for i in range(num_graphs):
#             # Nodes belonging to graph i
#             mask = (batch == i)
#             node_indices = torch.where(mask)[0]
#             num_nodes_in_graph = mask.sum().item()

#             # Extract z_i for graph i
#             z_i_graph = z_i[mask]  # [num_nodes_in_graph, zi_dim]

#             # Get reconstructed adjacency matrix A_hat for graph i
#             A_hat = A_hat_batch[i]  # [num_nodes_in_graph, num_nodes_in_graph]

#             # Convert dense matrix to edge_index and edge_weight
#             edge_index, edge_weight = dense_to_sparse(A_hat)

#             # GCN and MLP layers ?
#             X_hat_graph = self.node_decoder(z_i_graph, edge_index, edge_weight)
#             X_hat[mask] = X_hat_graph
        
        return A_hat_batch, X_hat