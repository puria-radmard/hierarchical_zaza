import torch
from torch import nn

class GraphVAELoss(nn.Module):
    """
    Computes the loss function for the HierarchicalGVAE.

    method args:
        A_hat_batch (Tensor): Reconstructed adjacency matrices [batch, num_nodes, num_nodes].
        # X_hat (Tensor): Reconstructed node features [num_nodes, node_feature_dim].
        data (Batch): Original data batch containing edge_index, x, and batch attributes.
        mu_A (Tensor): Mean of z_A [batch_size, za_dim].
        logvar_A (Tensor): Log variance of of z_A [batch_size, za_dim].
        mu_i (Tensor): Mean of z_i [num_nodes, zi_dim].
        logvar_i (Tensor): Log variance of z_i [num_nodes, zi_dim].
        device (torch.device): CPU/GPU.

    method returns:
        total_loss (Tensor): Total loss for the batch.
        adj_recon_term (Tensor): Reconstruction loss for adjacency matrices.
        # node_feature_recon_term (Tensor): Reconstruction loss for node features.
        kl_za_term (Tensor): KL divergence loss for graph-level latent variables.
        kl_zi_term (Tensor): KL divergence loss for node-level latent variables.
    """
    
    def __init__(self, 
                 device: torch.device, 
                 sparsity_weight: float = 1.0, 
                 beta_A: float = 1.0, 
                 beta_i: float = 1.0
                ):
        assert 0.0 < sparsity_weight <= 1.0
        self.sparsity_weight = sparsity_weight
        self.beta_A = beta_A
        self.beta_i = beta_i
        self.device = device
        super(GraphVAELoss, self).__init__()
        
    @staticmethod
    def kl_between_two_gaussians(m1, logvar1, m2, logvar2):
        """
        KL[ N(m1, logvar1) || N(m2, logvar2) ]
            i.e. both distributions parameterised by log of diagonal of covariance matrix
                 
        Shapes:
            m1: [batch, dim]
            logvar1: [batch, dim]
            m2: [batch, dim]
            logvar2: [batch, dim]
            
        Output: [batch], all elements > 0.0
        """
        det1 = logvar1.sum(-1).exp()   # [batch]
        det2 = logvar2.sum(-1).exp()   # [batch]
        det_term = det1 / det2         # [batch]
        
        d_term = m1.shape[1]
        
        trace_term = (logvar1 - logvar2).exp().sum(-1)   # [batch]
        
        mean_diff = m2 - m1
        inner_prod_term = (mean_diff.square() / logvar2.exp()).sum(-1) # [batch]
        
        return 0.5 * det_term - d_term + trace_term + inner_prod_term    
    
    def kl_zi_term(self, mu_i, logvar_i, z_a, all_B_mus, all_B_logvars):
        """
        Inputs:
            mu_i: [batch, num_nodes, dim_zi]
            logvar_i: [batch, num_nodes, dim_zi]
            z_a: [batch, dim_za]
            all_B_mus: [num_nodes, dim_zi, dim_za]
            all_B_logvars: [num_nodes, dim_zi, dim_za]
        
        Sum of each KL to node-wise prior:
            Sum_i KL[ N(mu_i, logvar) || N(...) ]
        """
        z_a = z_a.detach()
        N = all_B_logvars.shape[0]
        
        all_prior_means = torch.einsum('nfg,bg->bnf', all_B_mus, z_a).detach()         # [batch, num_nodes, dim_zi]
        all_prior_logvars = torch.einsum('nfg,bg->bnf', all_B_logvars, z_a).detach()   # [batch, num_nodes, dim_zi]
        
        cumulative_kls = 0.0  # [batch]
        
        for node_index in range(N):
            prior_mean = all_prior_means[:,node_index]      # [batch, dim_zi]
            prior_logvar = all_prior_logvars[:,node_index]  # [batch, dim_zi]
            node_kl = self.kl_between_two_gaussians(
                mu_i[:,node_index], logvar_i[:,node_index],
                prior_mean, prior_logvar)    # [batch]
            cumulative_kls = cumulative_kls + node_kl
            
        return cumulative_kls.mean()   # scalar

    def kl_za_term(self, mu_A, logvar_A):
        """
        Inputs:
            mu_A: [batch, dim_za]
            logvar_A: [batch, dim_za]
        
        KL to prior:
            KL[ N(mu, logvar) || N(O,I) ]
            scalar size
        """
        prior_mean = torch.zeros_like(mu_A, device = mu_A.device, dtype = mu_A.dtype)
        prior_logvar = torch.zeros_like(mu_A, device = mu_A.device, dtype = mu_A.dtype)
        return self.kl_between_two_gaussians(mu_A, logvar_A, prior_mean, prior_logvar).mean()
        
    def adj_recon_term(self, A_hat_batch, A_true):
        """
        Inputs:
            A_hat_batch of shape [batch, N, N]
                A_hat_batch[b,i,j] = p(A_b[i,j] = 1 | ...)
                
            A_true of shape [batch, N, N]
                A_true[b,i,j] = A_b[i,j] \in {0, 1}
                
        Cross entropy (elementwise):
            log( p(A)^A * (1-p(A))^(1-A) ) = Alogp(A) + (1-A)logp(1-A)
        """
        A_hat_batch[A_hat_batch == 0.0] = 1e-10
        A_hat_batch[A_hat_batch == 1.0] = 1. - 1e-10
        
        log_likelihood = (
            (A_true * A_hat_batch.log()) + 
            ((1.0 - A_true) * (1.0 - A_hat_batch).log())
        )
        
        log_likelihood[A_true == 0.0] = log_likelihood[A_true == 0.0] * self.sparsity_weight
            
        return log_likelihood.sum(-1).sum(-1).mean()
        
    def node_feature_recon_term(self, X_hat, X_true):
        return 0.0
    
    def forward(
        self, 
        A_hat_batch, A_true, X_hat, X_true, 
        mu_A, logvar_A, mu_i, logvar_i, z_a, 
        all_B_mus, all_B_logvars
    ):

        A_recon = self.adj_recon_term(A_hat_batch, A_true)
        X_recon = self.node_feature_recon_term(X_hat, X_true)
        KL_A = self.kl_za_term(mu_A, logvar_A)
        KL_i = self.kl_zi_term(mu_i, logvar_i, z_a, all_B_mus, all_B_logvars)
        
        return {
            'A_recon_loss': A_recon,
            'X_recon': X_recon,
            'KL_A': KL_A,
            'KL_i': KL_i,
            'negative_beta_ELBO': - (A_recon + X_recon + self.beta_A * KL_A + self.beta_i * KL_i)
        }
    
    
    