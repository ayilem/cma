import os
import torch 
import torch.nn as nn 


class Encoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, latent_dims)
        )

    def forward(self, x):       
        out = self.encoder(x)
    
        return out
    
class Decoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_input),
        )

    def forward(self, z):
        return self.decoder(z)


class MultimodalAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(MultimodalAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        
        return o
