import os
import torch 
import torch.nn as nn 


# class Encoder(nn.Module):
    
#     def __init__(self, n_input, latent_dims, n_hidden):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(n_input, n_hidden),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(n_hidden, n_hidden),
#         )

#         self.z_mean = nn.Linear(n_hidden, latent_dims)
#         self.z_log_var = nn.Linear(n_hidden, latent_dims)

#         self.kl = 0

#     def forward(self, x):       
#         out = self.encoder(x)
#         mu = self.z_mean(out)
#         logvar = self.z_log_var(out)
        
#         self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         # Reparametrization        
#         # std = torch.exp(0.5 * logvar)            
#         # eps = torch.randn_like(std)
#         # z = mu + eps * std
    
#         return z
    

# class Decoder(nn.Module):
    
#     def __init__(self, n_input, latent_dims, n_hidden):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dims, n_hidden),
#             #nn.BatchNorm1d(n_hidden),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(n_hidden, n_hidden),
#             #nn.BatchNorm1d(n_hidden),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(n_hidden, n_input),
#         )

#     def forward(self, z):
#         return self.decoder(z)
    

# class Discriminator(nn.Module):
    
#     def __init__(self, nz=128, n_hidden=1024, n_out=1):
#         super().__init__()
#         self.nz = nz
#         self.n_hidden = n_hidden
#         self.n_out = n_out

#         self.net = nn.Sequential(
#             nn.Linear(nz, n_hidden),
#             nn.ReLU(),
            
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
            
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
            
#             nn.Linear(n_hidden, n_out)
#         )

#     def forward(self, x):
#         return self.net(x)
    

#-------------------------------------------------------------------------

import torch
from torch import nn

class MultimodalVAE(nn.Module):
    def __init__(self, n_inputs1, n_inputs2, n_outputs, latent_dims, n_hiddens):
        super(MultimodalVAE, self).__init__()

        # Encodeurs pour les modalités Expression et Methylation
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

        # Décodeur pour la modalité Protein
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dims, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_outputs)
        )


    def forward(self, x1, x2):
        # Encode les modalités Expression et Methylation
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        # Combine les espaces latents
        z = torch.cat((z1, z2), dim=1)
        print(z.shape)
        # Décode pour générer la modalité Protein
        return self.decoder(z)