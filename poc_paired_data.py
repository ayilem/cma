import torch
from src.models_emma import VAE, Model 
from src.config import config
from src.dataset import get_paired_data
from torch.utils.data import DataLoader
from train_emma import CycledVAE, CoupledVAE

# Load the trained VAE models for all modalities
vae_model_modality1 = VAE(n_input = 131, n_hidden = 256)
vae_model_modality2 = VAE(n_input=367, n_hidden = 256) 
vae_model_modality3 = VAE(n_input=160, n_hidden = 256)  

# Load the trained weights for the VAE models
checkpoint_modality1 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint_vae_paired/vae_0_epoch_100.pth', map_location=torch.device('cpu'))
checkpoint_modality2 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint_vae_paired/vae_1_epoch_100.pth', map_location=torch.device('cpu'))
checkpoint_modality3 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint_vae_paired/vae_2_epoch_100.pth', map_location=torch.device('cpu'))

vae_model_modality1.load_state_dict(checkpoint_modality1)
vae_model_modality2.load_state_dict(checkpoint_modality2)
vae_model_modality3.load_state_dict(checkpoint_modality3)

vae = [vae_model_modality1, vae_model_modality2, vae_model_modality3]

# Load VAE via "Model" class to use "forward_vae" function
model= Model(config, vae)

from sklearn.preprocessing import StandardScaler

# Load Data
datasets = list()
datasets.extend(get_paired_data(suffix='10_equal'))

try : 
    pred = model.forward_vae(datasets, encoder_only=True)
    # print(len(pred[0]), type(pred[0]))
except : 
    print("not the same size ...")


# Fixer la graine pour Torch (pour la reproductibilité des résultats avec PyTorch)
torch.manual_seed(42)

##### MODALITY 1 #####

# Generate samples in modality 1
latent_modality1 = pred[0][:200]
# print("size : ", latent_modality1.shape)
generated_samples_modality1_from_1 = vae_model_modality1.decoder(latent_modality1)

# Decode the samples into modality 2
decoded_samples_modality2_from_1 = vae_model_modality2.decoder(latent_modality1)

# Decode the samples into modality 3
decoded_samples_modality3_from_1 = vae_model_modality3.decoder(latent_modality1)    


# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality1 = generated_samples_modality1_from_1.detach().numpy()
decoded_samples_modality2 = decoded_samples_modality2_from_1.detach().numpy()
decoded_samples_modality3 = decoded_samples_modality3_from_1.detach().numpy()

##### MODALITY 2 #####

# Generate samples in modality 2
latent_modality2 = pred[1][:200]
# print("size : ", latent_modality2.shape)
generated_samples_modality2_from_2 = vae_model_modality2.decoder(latent_modality2)

# Decode the samples into modality 2
decoded_samples_modality1_from_2 = vae_model_modality1.decoder(latent_modality2)
    
# Decode the samples into modality 3
decoded_samples_modality3_from_2 = vae_model_modality3.decoder(latent_modality2)  

import matplotlib.pyplot as plt

# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality2 = generated_samples_modality2_from_2.detach().numpy()
decoded_samples_modality1 = decoded_samples_modality1_from_2.detach().numpy()
decoded_samples_modality3 = decoded_samples_modality3_from_2.detach().numpy()


##### MODALITY 3 #####

# Generate samples in modality 3
latent_modality3 = pred[2][:200]
# print("size : ", latent_modality3.shape)
generated_samples_modality3_from_3 = vae_model_modality3.decoder(latent_modality3)

# Decode the samples into modality 1
decoded_samples_modality1_from_3 = vae_model_modality1.decoder(latent_modality3)
    
# Decode the samples into modality 3
decoded_samples_modality2_from_3 = vae_model_modality2.decoder(latent_modality3)

# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality2 = generated_samples_modality3_from_3.detach().numpy()
decoded_samples_modality1 = decoded_samples_modality1_from_3.detach().numpy()
decoded_samples_modality3 = decoded_samples_modality2_from_3.detach().numpy()


##########

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Liste des matrices
outputs = [generated_samples_modality1_from_1.detach().numpy(), decoded_samples_modality1_from_2.detach().numpy(),decoded_samples_modality1_from_3.detach().numpy(), 
           generated_samples_modality2_from_2.detach().numpy(), decoded_samples_modality2_from_1.detach().numpy(), decoded_samples_modality2_from_3.detach().numpy(), 
           generated_samples_modality3_from_3.detach().numpy(), decoded_samples_modality3_from_1.detach().numpy(), decoded_samples_modality3_from_2.detach().numpy()]

modalities = ['Modalité 1', 'Modalité 2', 'Modalité 3']

# Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, perplexity=10, n_iter = 1000, random_state = 42)


# # Affichage des résultats
# plt.figure(figsize=(15, 5))

# # Génération de la modalité 1 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 1)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 1 depuis {modalities[i]}')
# plt.title('Génération de Modalité 1 depuis Modalités 1, 2, 3')
# plt.legend()

# # Génération de la modalité 2 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 2)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i+3])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 2 depuis {modalities[i]}', marker='o')
# plt.title('Génération de Modalité 2 depuis Modalités 1, 2, 3')
# plt.legend()

# # Génération de la modalité 3 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 3)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i+6])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 3 depuis {modalities[i]}', marker='o')
# plt.title('Génération de Modalité 3 depuis Modalités 1, 2, 3')
# plt.legend()

# plt.tight_layout()
# plt.show()

# ### DISPLAY SCORES ###

import torch
import torch.nn as nn

criterion = nn.MSELoss()

# loss3_from_1 = criterion(decoded_samples_modality3_from_1, generated_samples_modality3_from_3)
# print("loss 3 from 1 : ", loss3_from_1.detach().numpy())
# loss2_from_1 = criterion(decoded_samples_modality2_from_1, generated_samples_modality2_from_2)
# print("loss 2 from 1 : ", loss2_from_1.detach().numpy())

# loss3_from_2 = criterion(decoded_samples_modality3_from_2, generated_samples_modality3_from_3)
# print("loss 3 from 2 : ", loss3_from_2.detach().numpy())
# loss1_from_2 = criterion(decoded_samples_modality1_from_2, generated_samples_modality1_from_1)
# print("loss 1 from 2 : ", loss1_from_2.detach().numpy())

# loss1_from_3 = criterion(decoded_samples_modality1_from_3, generated_samples_modality1_from_1)
# print("loss 1 from 3 : ", loss1_from_3.detach().numpy())
# loss2_from_3 = criterion(decoded_samples_modality2_from_3, generated_samples_modality2_from_2)
# print("loss 2 from 3 : ", loss2_from_3.detach().numpy())


# loss3_from_3 = criterion(datasets[2][0][:200], generated_samples_modality3_from_3)
# print("loss 3 from 3 : ", loss3_from_3.detach().numpy())
# loss3_from_2 = criterion(datasets[2][0][:200], decoded_samples_modality3_from_2)
# print("loss 3 from 2 : ", loss3_from_2.detach().numpy())
# loss3_from_2 = criterion(datasets[2][0][:200], decoded_samples_modality3_from_1)
# print("loss 3 from 1 : ", loss3_from_2.detach().numpy())
# print('--------------------')
# loss2_from_3 = criterion(datasets[1][0][:200], decoded_samples_modality2_from_3)
# print("loss 2 from 3 : ", loss2_from_3.detach().numpy())
# loss2_from_2 = criterion(datasets[1][0][:200], generated_samples_modality2_from_2)
# print("loss 2 from 2 : ", loss2_from_2.detach().numpy())
# loss2_from_1 = criterion(datasets[1][0][:200], decoded_samples_modality2_from_1)
# print("loss 2 from 1 : ", loss2_from_1.detach().numpy())
# print('--------------------')
# loss1_from_3 = criterion(datasets[0][0][:200], decoded_samples_modality1_from_3)
# print("loss 1 from 3 : ", loss1_from_3.detach().numpy())
# loss1_from_2 = criterion(datasets[0][0][:200],  decoded_samples_modality1_from_2)
# print("loss 1 from 2 : ", loss1_from_2.detach().numpy())
# loss1_from_1 = criterion(datasets[0][0][:200], generated_samples_modality1_from_1)
# print("loss 1 from 1 : ", loss1_from_1.detach().numpy())
# print('--------------------')


#### LOOP ON THE DATA ####

# model1= Model(config, [vae_model_modality1])
# model2= Model(config, [vae_model_modality2])
# model3= Model(config, [vae_model_modality3])

data1 = datasets[0][0][:200]
data2 = datasets[1][0][:200]
data3 = datasets[2][0][:200]

# pred1 = model1.forward_vae(data1, encoder_only=True)

#Next steps : different loss ? Puis train cette partie ? 
# Question : qu'est ce qu'on cherche à train ? Plutôt la partie décodeur de la modalité 3 puis encodeur de la modalité 3 ?
# Donc pour ça on fige les poids de la partie encodeur de la modalité 1 et décodeur de la modalité 1 ? Faire un schéma pour comprendre
# Attention on crée modèle et on utilise objet VAE en même temps ? Bizarre


# # Instantiate models
model1= Model(config, [vae_model_modality1])
model2= Model(config, [vae_model_modality2])
model3= Model(config, [vae_model_modality3])

# # Instantiate combined model
# cycled_model = CycledVAE(model1, model2)
# print(cycled_model.forward(data1))


from torch.utils.data import TensorDataset, DataLoader

# # # Train the combined model
# def train_cycle_vae(x, epoches, cycled_model, log_interval=10):
#     # Créer un TensorDataset avec uniquement les données de la modalité 1
#     x = torch.Tensor(x)  # Remplace modality1_data par tes données pour la modalité 1
#     dataset = TensorDataset(x)  # Remplace modality1_data par tes données pour la modalité 1
#     # Créer un DataLoader pour itérer sur les données par lots
#     batch_size = 32  # Choisis la taille de tes lots
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     optimizera = cycled_model.optimizera
#     optimizerb = cycled_model.optimizerb

#     for epoch in range(epoches):
#         for batch_idx, x in enumerate(train_loader):
#             x = torch.Tensor(x[0])
#             cycled_model.train()
#             optimizera.zero_grad()
#             optimizerb.zero_grad()
#             latent_1, output_1, latent_2, output_2 = cycled_model(x)
#             loss = criterion(x, output_2)  # Define your custom loss function
#             loss.backward()
#             optimizera.step()
#             optimizerb.step()
#             if batch_idx % log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(x), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader), loss.item() / len(x)))

# import torch.nn.functional as F

# # Train the combined model
# print("type data1 : ", type(data1))
# train_cycle_vae(data1, 100, cycled_model, log_interval=10)



# # Structure du code dans train_emma : 
# # def des 3 vae 
# # def des 9 loss
# # pour chaque vae munimodal : lancer en unimodal avec encod et decod
# # pour chaque vae paire : créer la paire (vae_1_from_2 etc) avec encodeur et décodeur

# # def une fonction train ici ?
# # boucle sur les epochs : 
# # calculer les loss, les sommer et train là dessus ?
# # En sortie je veux évolution des loss, val et test ?

# # Questions : 
# # - comment normaliser ? Revoir ça, demander comment multimodlaité gère ça
# # - créer ces modèles combinés : modèle à part entière comme pour les VAE ? Modifier du code source ?
# # - Ce qui a été modifié : datasets (get_paired_data), VAE (le forward)$
# # - Comment on fait pour les données de test ?
# # - Comment on fait pour les données de validation ?
# # - Pourquoi j'utilise parfois forward_vae, parfois forward, parfois decoder ?
# # - Voir si le croisement se fait bien (notamment voir si output2_1 et output1-2 sont bien définis ou lancent juste un VAE simple)
# # - puis-je créer une loss générale ? 

# print("shape input: ", data1.shape)
# output1_1 = vae_model_modality1.forward(data1)
# print("shape 1-1 : ", output1_1.shape)
# output1_3 = vae_model_modality1.forward(data1, vae_model_modality3)
# print("shape 1-3 : ", output1_3.shape)

def train_coupled_vae( datasets, epoches, coupled_vae_model, log_interval=10):
    # Créer un TensorDataset qui fusionne les données des 3 modalités
    datasets = [torch.Tensor(datasets[i][0]) for i in range(3)]
    datasets = TensorDataset(*datasets)
    batch_size = 32  # Choisis la taille de tes lots
    # obtenir le tensor dataset
    # print(type(data), len(data), len(data[0]))
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    print(" dimensions : ", len(train_loader), len(train_loader.dataset), len(train_loader.dataset[0]))
    optimizer1 = coupled_vae_model.optimizer1
    optimizer2 = coupled_vae_model.optimizer2
    optimizer1_2 = coupled_vae_model.optimizer1_2
    optimizer2_1 = coupled_vae_model.optimizer2_1
    # optimizer = torch.optim.Adam(coupled_vae_model.parameters(), lr=0.001)

    for epoch in range(epoches):
        for batch_idx, (x1, x2, x3) in enumerate(train_loader):
            x1 = torch.Tensor(x1)
            x2 = torch.Tensor(x2)
            coupled_vae_model.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer1_2.zero_grad()
            optimizer2_1.zero_grad()

            output1,output2, output1_2, output2_1 = coupled_vae_model(x1, x2)

            loss1 = criterion(x1, output1)
            loss2 = criterion(x2, output2)
            loss1_2 = criterion(output1, output2_1)
            loss2_1 = criterion(output2, output1_2)
            loss = loss1 + loss2 + loss1_2 + loss2_1
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer1_2.step()
            optimizer2_1.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x1), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(x1)))


# Instantiate VAEs
vae1 = VAE(n_input = 131, n_hidden = 256)
vae2 = VAE(n_input=367, n_hidden = 256) 
vae3 = VAE(n_input=160, n_hidden = 256)
vae2_1 = VAE(n_input = 131, n_hidden = 256, other_m=vae2)
vae1_2 = VAE(n_input=367, n_hidden = 256, other_m=vae1)

# Instantiate models
model1= Model(config, [vae1])
model2= Model(config, [vae2])
model3= Model(config, [vae3])
model1_2= Model(config, [vae1_2])
model2_1= Model(config, [vae2_1])

coupled_vae_model = CoupledVAE(model1, model2, model3, model1_2, model2_1)
# print(" forward : ", coupled_vae_model.forward(data1, data2))
train_coupled_vae( datasets, 10, coupled_vae_model, log_interval=10)