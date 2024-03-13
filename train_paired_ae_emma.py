import argparse
import os
import torch
from torch.utils.data import DataLoader
from model_paired_ae_emma import MultimodalAE, Encoder, Decoder
from src.dataset import generate_datasets


script_dir = os.path.dirname(__file__)


#-------------------------------------------------

from torch.utils.data import DataLoader
from src.dataset import generate_datasets
import torch.nn as nn
import torch
from src.config import config

# Génération des ensembles de données
train_datasets = generate_datasets(suffix='5_diff', type='paired', train=True, test=False)
test_datasets = generate_datasets(suffix='5_diff', type='paired', train=False, test=True)


# Création des chargeurs de données
train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets]
test_loaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets]


# Obtenez les dimensions d'entrée à partir des ensembles de données
n_inputs1 = train_datasets[0][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 1
n_inputs2 = train_datasets[1][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 2
n_inputs3 = train_datasets[2][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 3

# print(train_datasets[0][0][0].shape, train_datasets[1][0][0].shape, train_datasets[2][0][0].shape)
                              
# Obtenez les dimensions de l'espace latent et du réseau caché à partir de config.py
latent_dims = 20   
n_hiddens = 256 

# Création du modèle
encoder1 = Encoder(n_inputs1, latent_dims, n_hiddens)
encoder2 = Encoder(n_inputs2, latent_dims, n_hiddens)

z1 = encoder1(train_datasets[0][0][0])
z2 = encoder2(train_datasets[1][0][0])

decoder1 = Decoder(n_inputs1, latent_dims, n_hiddens)
decoder2 = Decoder(n_inputs2, latent_dims, n_hiddens)

model1 = MultimodalAE(encoder=encoder1, decoder=decoder1)
model2 = MultimodalAE(encoder=encoder2, decoder=decoder2)
model1_2 = MultimodalAE(encoder=encoder1, decoder=decoder2)
model2_1 = MultimodalAE(encoder=encoder2, decoder=decoder1)

# Choix de la fonction de perte et de l'optimiseur
critere = nn.MSELoss()
optimizer1 = torch.optim.Adam(model1.parameters())
optimizer2 = torch.optim.Adam(model2.parameters())
optimizer1_2 = torch.optim.Adam(model1_2.parameters())
optimizer2_1 = torch.optim.Adam(model2_1.parameters())


# Entraînement du modèle
num_epochs = 100
for epoch in range(num_epochs):
    model1.train()
    model2.train()
    model1_2.train()
    model2_1.train()
    for (x1, _), (x2, _), (y, _) in zip(*train_loaders):

        if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
            continue

       # Forward pass
        o1, o2, o1_2, o2_1 = model1(x1), model2(x2), model1_2(x1), model2_1(x2)
        loss = 0*critere(o1, x1) + 0*critere(o2, x2) + critere(o1_2, x2) + 0*critere(o2_1, x1)
        # Backward and optimize
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer1_2.zero_grad()
        optimizer2_1.zero_grad()

        loss.backward()

        optimizer1_2.step()
        # optimizer2_1.step()
        # optimizer1.step()
        # optimizer2.step()

    print(f' Epoch {epoch} -> loss totale :', loss.item(), " loss1 :", critere(o1, x1).item(), " loss2 :", critere(o2, x2).item(), " loss1_2 :", critere(o1_2, x2).item(), " loss2_1 :", critere(o2_1, x1).item())

    #     # Vérification des paramètres partagés entre les modèles
    # print("Encoder1 parameters model1:")
    # for name, param in model1.encoder.named_parameters():
    #     # print(name, param.data)

    # print("Encoder1 parameters model1_2:")
    # for name, param in model1_2.encoder.named_parameters():
    #     print(name, param.data)

    # print("\nDecoder2 parameters model2:")
    # for name, param in model2.decoder.named_parameters():
    #     print(name, param.data)
        
    # print("\nDecoder2 parameters model1_2:")
    # for name, param in model1_2.decoder.named_parameters():
    #     print(name, param.data)


model1.eval()
model2.eval()
model1_2.eval()
model2_1.eval()
with torch.no_grad():
    total_loss = 0
    for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
        o1, o2, o1_2, o2_1 =  model1(x1), model2(x2), model1_2(x1), model2_1(x2)
        loss = critere(o1, x1) + critere(o2, x2) + critere(o1_2, x2) + critere(o2_1, x1)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loaders[0])}')
    print('loss1 :', critere(o1, x1).item(), 'loss2 :', critere(o2, x2).item(), 'loss1_2 :', critere(o1_2, x2).item(), 'loss2_1 :', critere(o2_1, x1).item())
