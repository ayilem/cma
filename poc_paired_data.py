import torch
from src.models_emma import VAE, Model 
from src.config import config
from src.dataset import get_paired_data
from torch.utils.data import DataLoader

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
print("size : ", latent_modality1.shape)
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
print("size : ", latent_modality2.shape)
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
print("size : ", latent_modality3.shape)
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


# Affichage des résultats
plt.figure(figsize=(15, 5))

# Génération de la modalité 1 à partir des modalités 1, 2, 3
plt.subplot(1, 3, 1)
for i in range(3):
    reduced_data = tsne.fit_transform(outputs[i])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 1 depuis {modalities[i]}')
plt.title('Génération de Modalité 1 depuis Modalités 1, 2, 3')
plt.legend()

# Génération de la modalité 2 à partir des modalités 1, 2, 3
plt.subplot(1, 3, 2)
for i in range(3):
    reduced_data = tsne.fit_transform(outputs[i+3])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 2 depuis {modalities[i]}', marker='o')
plt.title('Génération de Modalité 2 depuis Modalités 1, 2, 3')
plt.legend()

# Génération de la modalité 3 à partir des modalités 1, 2, 3
plt.subplot(1, 3, 3)
for i in range(3):
    reduced_data = tsne.fit_transform(outputs[i+6])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité 3 depuis {modalities[i]}', marker='o')
plt.title('Génération de Modalité 3 depuis Modalités 1, 2, 3')
plt.legend()

plt.tight_layout()
plt.show()

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


loss3_from_3 = criterion(datasets[2][0][:200], generated_samples_modality3_from_3)
print("loss 3 from 3 : ", loss3_from_3.detach().numpy())
loss3_from_2 = criterion(datasets[2][0][:200], decoded_samples_modality3_from_2)
print("loss 3 from 2 : ", loss3_from_2.detach().numpy())
loss3_from_2 = criterion(datasets[2][0][:200], decoded_samples_modality3_from_1)
print("loss 3 from 1 : ", loss3_from_2.detach().numpy())
print('--------------------')
loss2_from_3 = criterion(datasets[1][0][:200], decoded_samples_modality2_from_3)
print("loss 2 from 3 : ", loss2_from_3.detach().numpy())
loss2_from_2 = criterion(datasets[1][0][:200], generated_samples_modality2_from_2)
print("loss 2 from 2 : ", loss2_from_2.detach().numpy())
loss2_from_1 = criterion(datasets[1][0][:200], decoded_samples_modality2_from_1)
print("loss 2 from 1 : ", loss2_from_1.detach().numpy())
print('--------------------')
loss1_from_3 = criterion(datasets[0][0][:200], decoded_samples_modality1_from_3)
print("loss 1 from 3 : ", loss1_from_3.detach().numpy())
loss1_from_2 = criterion(datasets[0][0][:200],  decoded_samples_modality1_from_2)
print("loss 1 from 2 : ", loss1_from_2.detach().numpy())
loss1_from_1 = criterion(datasets[0][0][:200], generated_samples_modality1_from_1)
print("loss 1 from 1 : ", loss1_from_1.detach().numpy())
print('--------------------')


#### LOOP ON THE DATA ####

def loop_on_two_modalities(int1, int2):
    # First step : generate output modality int2 from input modality int1
    latent_modality = pred[int1][:200]
    decoded_samples_modality3_from_1 = vae[int2-1].decoder(latent_modality)
    return "ok"