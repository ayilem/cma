import os
import torch 
import torch.nn as nn 


#-------------------------------------------------------------------------

import torch
from torch import nn

class MultimodalAE(nn.Module):
    def __init__(self, n_inputs1, n_inputs2, n_outputs, latent_dims, n_hiddens):
        super(MultimodalAE, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(n_inputs1, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, latent_dims)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(n_inputs2, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, latent_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dims, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_outputs)
        )


    def forward(self, x1, x2):

        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        z = torch.cat((z1, z2), dim=1)
        print(z.shape)

        return self.decoder(z)