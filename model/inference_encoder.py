import torch
from torch import nn

from torch_scatter import scatter_softmax, scatter_add
from torch_geometric.nn import GCNConv



class inference_model(nn.Module):
    def __init__(self, hparams: dict, node_specific_loading: bool = True):
        super(inference_model, self).__init__()
        self.hparams = hparams
        self.node_specific_loading = node_specific_loading

        self.num_nodes_per_graph = hparams['num_nodes_per_graph']   # N        
        self.gcn_hidden_dim = hparams['gcn_hidden_dim']             # Nh
        self.za_dim = hparams['za_dim']                             # G
        self.zi_dim = hparams['zi_dim']                             # F

        # Graph convolutional layers (3-hop)
        self.conv1 = GCNConv(hparams['node_feature_dim'], self.gcn_hidden_dim)
        self.conv2 = GCNConv(self.gcn_hidden_dim, self.gcn_hidden_dim)
        self.conv3 = GCNConv(self.gcn_hidden_dim, self.gcn_hidden_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(self.gcn_hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.gcn_hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.gcn_hidden_dim)
        
        # Attention mechanism to compute attention scores per node
        # φ = φ_1 * h1 + φ_2 * h2 + φ_bias                   [num_nodes,1]
        # attn_weights = softmax(φ)                          [num_nodes,1]
        # φ_weighted = atten_weights * h3                    [num_nodes, gcn_hidden_dim]
        # h_graph = sum(φ_weighted)                          [batch_size, gcn_hidden_dim]
        self.phi_h1 = nn.Linear(self.gcn_hidden_dim, 1)
        self.phi_h2 = nn.Linear(self.gcn_hidden_dim, 1, bias=False)

        # Linear layers to compute μ_A and log Σ_A
        self.fc_mu_A = nn.Linear(self.gcn_hidden_dim, self.za_dim)
        self.fc_logvar_A = nn.Linear(self.gcn_hidden_dim, self.za_dim)
        
        # Linear layers for node-level latent variables s_i (linear combination of h and z_a)
        # s_i = W_h h_i + b_h + W_A z_A + b_A
        self.W_h = nn.Linear(self.gcn_hidden_dim, self.zi_dim)           # W_h: maps h_i to s_i
        
        if self.node_specific_loading:         
            # Node-specific weight matrices for transforming z_A when computing s_i
            # s_i = W_h h_i + b_h + W_A_i z_A + b_A
            self.W_A = nn.Parameter(torch.randn(self.num_nodes_per_graph, self.za_dim, self.zi_dim))
        else:
            self.W_A = nn.Linear(self.za_dim, self.zi_dim, bias=False)   # W_A: maps z_A to s_i
        
        # Linear layers to compute μ_i and log Σ_i
        self.fc_mu_i = nn.Linear(self.zi_dim, self.zi_dim)
        self.fc_logvar_i = nn.Linear(self.zi_dim, self.zi_dim)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
            edge_index (Tensor): Edge indices.

        Returns:
            z_i (Tensor): Sampled node-level latent variables.
            mu_i (Tensor): Mean of the node-level latent variables.
            logvar_i (Tensor): Log variance of the node-level latent variables.
            z_A (Tensor): Sampled graph-level latent variable.
            mu_A (Tensor): Mean of the graph-level latent variable.
            logvar_A (Tensor): Log variance of the graph-level latent variable.
        """      
        # ===== 1. Node hidden states with GCN =====
        h1 = torch.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = torch.relu(self.bn2(self.conv2(h1, edge_index)))
        h3 = torch.relu(self.bn3(self.conv3(h2, edge_index)))  # [num_nodes, gcn_hidden_dim] 
          
        # ===== 2. Attention Pooling to aggregate node-level hidden state to graph-level =====
        # Options: 1) global pooling; 2) hierarchical (local) pooling
        # Review hierarchical pooling, e.g., Self-Attention Graph Pooling (Lee et al., 19')
        # Global pooling implemented below
        
        # Compute attention scores for each node at h1 and h2
        phi = self.phi_h1(h1) + self.phi_h2(h2)                # [num_nodes, 1]
        attn_scores = phi.squeeze(-1)                          # [num_nodes]

        # Compute attention weights using softmax over nodes in the same graph
        attn_weights = scatter_softmax(attn_scores, batch)     # [num_nodes]
        # Weight node embeddings
        attn_weights = attn_weights.unsqueeze(-1)              # [num_nodes, 1]
        phi_weighted = h3 * attn_weights                       # [num_nodes, gcn_hidden_dim]
        # Sum over nodes per graph to get graph-level representation
        h_graph = scatter_add(phi_weighted, batch, dim=0)      # [batch_size, gcn_hidden_dim]
        
        # h_graph = global_mean_pool(h, batch) # simple global mean pooling

        # ===== 3. Z_A; q(Z_A∣X,A) =====
        # Compute μ_A and log Σ_A
        mu_A = self.fc_mu_A(h_graph)            # [batch_size, za_dim] - Mean μ_h
        logvar_A = self.fc_logvar_A(h_graph)    # [batch_size, za_dim] - Log variance log Σ_h
        
        # Reparameterization trick for Z_A
        std_A = torch.exp(0.5 * logvar_A)
        eps_A = torch.randn_like(std_A)
        z_A = mu_A + eps_A * std_A              # [batch_size, za_dim]
        
        # ===== 4. Z_i; q(Z_i|Z_A,X,A) =====
        batch_size = batch.max().item() + 1
        z_A_expanded = z_A[batch]               # [num_nodes, za_dim]
        
        if self.node_specific_loading:
            
            # Reshape z_A to [batch_size, num_nodes_per_graph, za_dim] for batch matrix multiplication
            z_A_reshaped = z_A_expanded.view(batch_size, self.num_nodes_per_graph, self.za_dim).transpose(0, 1)
            # Perform batch matrix multiplication
            # W_A: [num_nodes_per_graph, za_dim, zi_dim]; ngf
            # z_A_reshaped: [num_nodes_per_graph, batch_size, za_dim]; nbg
            # z_A_transformed: [num_nodes_per_graph, batch_size, zi_dim]; nbf
            z_A_transformed = torch.einsum('ngf,nbg->nbf', self.W_A, z_A_reshaped)
            # Reshape to [num_nodes_per_graph * batch_size, zi_dim]
            z_A_transformed = z_A_transformed.transpose(0, 1).reshape(self.num_nodes_per_graph * batch_size, self.zi_dim)
            
            # Linear combination: s_i = W_h h_i + W_A_i z_A
            s_i = self.W_h(h3) + z_A_transformed               # [num_nodes, zi_dim]
            
        else: 
            # Linear combination: s_i = W_h h_i + W_A z_A 
            s_i = self.W_h(h3) + self.W_A(z_A_expanded)        # [num_nodes, zi_dim]
        
        # Compute μ_i and log Σ_i
        mu_i = self.fc_mu_i(s_i)                # [num_nodes, zi_dim]
        logvar_i = self.fc_logvar_i(s_i)        # [num_nodes, zi_dim]
        
        # Reparameterization trick for Z_i
        std_i = torch.exp(0.5 * logvar_i)
        eps_i = torch.randn_like(std_i)
        z_i = mu_i + eps_i * std_i              # [num_nodes, zi_dim]
        
        return z_i, mu_i, logvar_i, z_A, mu_A, logvar_A