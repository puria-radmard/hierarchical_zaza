
def train_epoch(encoder, decoder, optimizer, loss_fn, train_loader, all_B_mus, all_B_logvars):
    encoder.train()
    decoder.train()
    
    epoch_tracking = {
        "A_recon_loss": [],
        "X_recon": [],
        "KL_A": [],
        "KL_i": [],
        "negative_beta_ELBO": [],
    }
        
    for batch in train_loader:
        
        optimizer.zero_grad()
        num_graphs = batch.num_graphs
        num_nodes_per_graph = batch.num_nodes // batch.num_graphs
        A_batch = torch.stack([data.adj for data in batch.to_data_list()], dim=0)  # [num_graphs, node_dim, node_dim]
        X_batch = batch.x # [num_nodes, x_dim]
        
        # inference model
        z_i, mu_i, logvar_i, z_A, mu_A, logvar_A = encoder(batch.x, batch.edge_index, batch.batch)
        # generative model
        A_hat_batch, X_hat = decoder(z_i, batch.batch)
        
        losses_dict = loss_fn(
            A_hat_batch, A_batch, X_hat, X_batch, 
            mu_A, logvar_A, mu_i, logvar_i, z_A, 
            all_B_mus, all_B_logvars
        )

        losses_dict['negative_beta_ELBO'].backward()
        optimizer.step()
        
        for k in losses_dict.keys():
            epoch_tracking[k].append(losses_dict[k].item())
    
    return epoch_tracking
