import os
import torch 
import torch.nn as nn 


class Encoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
        )

        self.z_mean = nn.Linear(n_hidden, latent_dims)
        self.z_log_var = nn.Linear(n_hidden, latent_dims)

        self.kl = 0

    def forward(self, x):       
        out = self.encoder(x)
        mu = self.z_mean(out)
        logvar = self.z_log_var(out)
        
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Reparametrization        
        std = torch.exp(0.5 * logvar)            
        eps = torch.randn_like(std)
        z = mu + eps * std
    
        return z
    

class Decoder(nn.Module):
    
    def __init__(self, n_input, latent_dims, n_hidden):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(n_hidden, n_input),
        )

    def forward(self, z):
        return self.decoder(z)
    

class Discriminator(nn.Module):
    
    def __init__(self, nz=128, n_hidden=1024, n_out=1):
        super().__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        return self.net(x)
    

# V1 

# class MultimodalVAE(nn.Module):
#     def __init__(self, modal1_latent_dims, modal2_latent_dims, output_dims):
#         super().__init__()
#         self.modal1_latent_dims = modal1_latent_dims
#         self.modal2_latent_dims = modal2_latent_dims
#         self.output_dims = output_dims

#         self.modal1_encoder = Encoder(n_input=modal1_input_dims, latent_dims=modal1_latent_dims, n_hidden=hidden_dims)
#         self.modal2_encoder = Encoder(n_input=modal2_input_dims, latent_dims=modal2_latent_dims, n_hidden=hidden_dims)
#         self.decoder = Decoder(n_input=output_dims, latent_dims=modal1_latent_dims + modal2_latent_dims, n_hidden=hidden_dims)

#     def forward(self, modal1_input, modal2_input):
#         modal1_latent = self.modal1_encoder(modal1_input)
#         modal2_latent = self.modal2_encoder(modal2_input)
#         combined_latent = torch.cat((modal1_latent, modal2_latent), dim=1)
#         output = self.decoder(combined_latent)
#         return output


# ----------------------------------
    
# V2
    
import torch
import torch.nn as nn




import torch
import torch.nn as nn

# class MultimodalVAE(nn.Module):
#     def __init__(self, n_inputs1, n_inputs2, latent_dims, n_hiddens):
#         super().__init__()

#         # Créez un encodeur pour chaque dimension d'entrée de chaque modalité
#         n_hidden = 256
#         n_hiddens = [256, 256, 256]
#         self.encoders1 = [Encoder(n_inputs1, latent_dims, n_hidden)]
#         self.encoders2 = [Encoder(n_inputs2, latent_dims, n_hidden)]

#         # Définissez le décodeur
#         self.decoder = Decoder(latent_dims, n_inputs1, n_hiddens[0])  # Assurez-vous de définir les dimensions d'entrée et de sortie correctement

#     def encode(self, x):
#         # Assurez-vous que 'x' est une liste de tensors
#         if not isinstance(x, list):
#             x = [x]

#         # Décompressez les tuples en deux listes distinctes de tenseurs
#         tensors1 = [t[0] for t in x]
#         tensors2 = [t[1] for t in x]

#         # Passez chaque tensor à travers son encodeur correspondant
#         encoded1 = [encoder(x_i) for x_i, encoder in zip(tensors1, self.encoders1)]
#         encoded2 = [encoder(x_i) for x_i, encoder in zip(tensors2, self.encoders2)]

#         # Concaténez les deux listes encodées
#         encoded = encoded1 + encoded2

#         return encoded

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#     # Assurez-vous que 'x' est une liste de tuples de tensors
#         if not isinstance(x, list):
#            x = [x]

#     # Décompressez les tuples en deux listes distinctes de tenseurs
#         tensors1 = [t[0] for t in x]
#         tensors2 = [t[1] for t in x]

#     # Encode
#         encoded1 = self.encode(tensors1)  # Utilisez la méthode encode pour obtenir la sortie de self.shared_layers
#         encoded2 = self.encode(tensors2)  # Utilisez la méthode encode pour obtenir la sortie de self.shared_layers

#         z_mean1, z_log_var1 = self.z_mean(encoded1), self.z_log_var(encoded1)
#         z_mean2, z_log_var2 = self.z_mean(encoded2), self.z_log_var(encoded2)

#         z1 = self.reparameterize(z_mean1, z_log_var1)
#         z2 = self.reparameterize(z_mean2, z_log_var2)

#     # Combine the latent vectors
#         z = torch.cat((z1, z2), dim=1)

#     # Decode
#         return self.decode(z), z, z_mean1, z_log_var1, z_mean2, z_log_var2








# def encode(self, x):
    # # Convertissez 'x' en un tensor PyTorch si ce n'est pas déjà le cas
    #     if not isinstance(x, torch.Tensor):
    #          x = torch.tensor(x)

    #     encoded = [encoder(x) for encoder in self.encoders]

    #     encoded = torch.cat(encoded, dim=1)
    #     x_concat = torch.cat(encoded, dim=1)  # Concatenate the tensors in the list along dimension 1
    #     return self.shared_layers(x_concat)
    
# Dans models_mel.py, méthode encode











#----------------------------------------------


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