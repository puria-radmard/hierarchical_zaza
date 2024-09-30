import torch
from torch import optim

from utils.data import NodeReorder
from utils.training import train_epoch
from model import Encoder, Decoder, LossFunction

import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MNISTSuperpixels

from matplotlib import pyplot as plt


#### GENERATE DATA
# laplacian positional embed + node reorder along y = x
transform = T.Compose([
    T.AddLaplacianEigenvectorPE(k=5, attr_name='laplacian_eigen_pe'),
    NodeReorder()
])

mnist_path = '/Users/subat/Desktop/datan/hierarchical gAVE/model dev/mnist_superpixel'
train_transformed = MNISTSuperpixels(root=mnist_path, transform=transform)
test_transformed = MNISTSuperpixels(root=mnist_path, train=False, transform=transform)


#### DEFINE DATALOADER
batch_size = 32
num_nodes_per_graph = 75

train_loader = DataLoader(train_transformed, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_transformed, batch_size=batch_size, shuffle=False)



#### DEFINE MODEL, LOSS FUNCTION, and regularisation priors
hparams = {
    'num_nodes_per_graph': num_nodes_per_graph,
    'node_feature_dim': train_transformed.num_features,  # Number of node features
    'gcn_hidden_dim': 64,  
    'za_dim': 8,          # Dimension of graph-level latent variable z_A
    'zi_dim': 4,          # Dimension of node-level latent variable z_i
    'attention_dim': 16
}
encoder = Encoder(hparams, node_specific_loading=True)
decoder = Decoder(hparams)
loss_func = LossFunction(device = 'cpu')

B_mus_linear_condtional_dependence = torch.randn(
    hparams['num_nodes_per_graph'], hparams['zi_dim'], hparams['za_dim'], 
)
B_logvars_linear_condtional_dependence = torch.randn(
    hparams['num_nodes_per_graph'], hparams['zi_dim'], hparams['za_dim'], 
)


#### INITIALISE TRAINING
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(parameters, lr=0.05, weight_decay=0.01)  # weight decay for regularization
# Define dependece of priors over zi conditioned on za
            # all_B_mus: [num_nodes, dim_zi, dim_za]
            # all_B_logvars: [num_nodes, dim_zi, dim_za]
num_epochs = 10
tracking = {
    "A_recon_loss": [],
    "X_recon": [],
    "KL_A": [],
    "KL_i": [],
    "negative_beta_ELBO": [],
}


#### RUN TRAINING AND PERIODICALLY PLOT
for _ in range(num_epochs):
    
    epoch_tracking = train_epoch(
        encoder, decoder, 
        optimizer, loss_func, 
        train_loader, 
        B_mus_linear_condtional_dependence,
        B_logvars_linear_condtional_dependence
    )

    plt.clf()

    for k in epoch_tracking.keys():
        tracking[k].extend(epoch_tracking[k])
    
        plt.plot(tracking[k], label = k)
        
        plt.savefig('training_attempt.png')
