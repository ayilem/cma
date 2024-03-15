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
encoder3 = Encoder(n_inputs3, latent_dims, n_hiddens)

z1 = encoder1(train_datasets[0][0][0])
z2 = encoder2(train_datasets[1][0][0])
z3 = encoder3(train_datasets[2][0][0])

decoder1 = Decoder(n_inputs1, latent_dims, n_hiddens)
decoder2 = Decoder(n_inputs2, latent_dims, n_hiddens)
decoder3 = Decoder(n_inputs3, latent_dims, n_hiddens)

model1 = MultimodalAE(encoder=encoder1, decoder=decoder1)
model2 = MultimodalAE(encoder=encoder2, decoder=decoder2)
model1_2 = MultimodalAE(encoder=encoder1, decoder=decoder2)
model2_1 = MultimodalAE(encoder=encoder2, decoder=decoder1)
model3 = MultimodalAE(encoder=encoder3, decoder=decoder3)
model1_3 = MultimodalAE(encoder=encoder1, decoder=decoder3)
model3_1 = MultimodalAE(encoder=encoder3, decoder=decoder1)
model2_3 = MultimodalAE(encoder=encoder2, decoder=decoder3)
model3_2 = MultimodalAE(encoder=encoder3, decoder=decoder2)

# Choix de la fonction de perte et de l'optimiseur
critere = nn.MSELoss()
lr = 0.05
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
optimizer1_2 = torch.optim.Adam(model1_2.parameters(), lr=lr)
optimizer2_1 = torch.optim.Adam(model2_1.parameters(), lr=lr)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr)
optimizer1_3 = torch.optim.Adam(model1_3.parameters(), lr=lr)
optimizer3_1 = torch.optim.Adam(model3_1.parameters(), lr=lr)



num_epochs = 100

# Consolidate models and optimizers
models = [model1_3, model3_1]
optimizers = [optimizer1_3, optimizer3_1]

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=20, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    for model, optimizer in zip(models, optimizers):
        model.train()
        for (x1, _), (x2, _), (x3, y) in zip(*train_loaders):
            if x1.size(0) != 32 or x3.size(0) != 32 or y.size(0) != 32:
                continue
            optimizer.zero_grad()
            o1_3, o3_1 = model(x1), model(x3)
            loss = critere(o1_3, x3) + critere(o3_1, x1)
            loss.backward()
            optimizer.step()
        scheduler.step()


# # Entraînement du modèle
# num_epochs = 100
# for epoch in range(num_epochs):
#     model1.train()
#     model3.train()
#     model1_3.train()
#     model3_1.train()
#     for (x1, _), (x2, _), (x3, y) in zip(*train_loaders):

#         if x1.size(0) != 32 or x3.size(0) != 32 or y.size(0) != 32:
#             continue

#        # Forward pass
#         o1, o3, o1_3, o3_1 = model1(x1), model3(x3), model1_3(x1), model3_1(x3)
#         loss = 0*critere(o1, x1) + 0*critere(o3, x3) + critere(o1_3, x3) + critere(o3_1, x1)
#         # loss = critere(o1, x1) + critere(o3, x3) + critere(o1_3, x3) + critere(o3_1, x1)
#         # Backward and optimize
#         # optimizer1.zero_grad()
#         # optimizer3.zero_grad()
#         optimizer1_3.zero_grad()
#         optimizer3_1.zero_grad()

#         loss.backward()

#         optimizer1_3.step()
#         optimizer3_1.step()
#         # optimizer1.step()
#         # optimizer3.step()

#     print(f' Epoch {epoch} -> loss totale :', loss.item(), " loss1 :", critere(o1, x1).item(), " loss3 :", critere(o3, x3).item(), " loss1_3 :", critere(o1_3, x3).item(), " loss3_1 :", critere(o3_1, x1).item())

# # model1.eval()
# # model3.eval()
# model1_3.eval()
# model3_1.eval()
# with torch.no_grad():
#     total_loss = 0
#     for (x1, _), (x2, _), (x3, y) in zip(*test_loaders):
#         o1, o3, o1_3, o3_1 = model1(x1), model3(x3), model1_3(x1), model3_1(x3)
#         loss = critere(o1, x1) + critere(o3, x3) + critere(o1_3, x3) + critere(o3_1, x1)
#         total_loss += loss.item()
#     print(f'Test Loss: {total_loss / len(test_loaders[0])}')
#     print(" loss1 :", critere(o1, x1).item(), " loss3 :", critere(o3, x3).item(), " loss1_3 :", critere(o1_3, x3).item(), " loss3_1 :", critere(o3_1, x1).item())



# # Entraînement du modèle
# num_epochs = 100
# for epoch in range(num_epochs):
#     model1.train()
#     model2.train()
#     model1_2.train()
#     model2_1.train()
#     for (x1, _), (x2, _), (y, _) in zip(*train_loaders):

#         if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
#             continue

#        # Forward pass
#         o1, o2, o1_2, o2_1 = model1(x1), model2(x2), model1_2(x1), model2_1(x2)
#         loss = critere(o1, x1) + critere(o2, x2) + critere(o1_2, x2) + critere(o2_1, x1)
#         # loss = 0*critere(o1, x1) + 0*critere(o2, x2) + critere(o1_2, x2) + 0*critere(o2_1, x1)
#         # Backward and optimize
#         optimizer1.zero_grad()
#         optimizer2.zero_grad()
#         optimizer1_2.zero_grad()
#         optimizer2_1.zero_grad()

#         loss.backward()

#         optimizer1_2.step()
        # optimizer2_1.step()
        # optimizer1.step()
        # optimizer2.step()

#     print(f' Epoch {epoch} -> loss totale :', loss.item(), " loss1 :", critere(o1, x1).item(), " loss2 :", critere(o2, x2).item(), " loss1_2 :", critere(o1_2, x2).item(), " loss2_1 :", critere(o2_1, x1).item())

# model1.eval()
# model2.eval()
# model1_2.eval()
# model2_1.eval()
# with torch.no_grad():
#     total_loss = 0
#     for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
#         o1, o2, o1_2, o2_1 =  model1(x1), model2(x2), model1_2(x1), model2_1(x2)
#         loss = critere(o1, x1) + critere(o2, x2) + critere(o1_2, x2) + critere(o2_1, x1)
#         total_loss += loss.item()
#     print(f'Test Loss: {total_loss / len(test_loaders[0])}')
#     print('loss1 :', critere(o1, x1).item(), 'loss2 :', critere(o2, x2).item(), 'loss1_2 :', critere(o1_2, x2).item(), 'loss2_1 :', critere(o2_1, x1).item())
