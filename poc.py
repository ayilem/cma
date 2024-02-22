import torch
from src.models import VAE  # Make sure to replace 'src.models' with the actual module path
from src.config import config

# Load the trained VAE models for both modalities
vae_model_modality1 = VAE(n_input = 131, n_hidden = 256)  # Replace with the actual VAE model for modality 1
vae_model_modality2 = VAE(n_input=367, n_hidden = 256)  # Replace with the actual VAE model for modality 2
vae_model_modality3 = VAE(n_input=160, n_hidden = 256)  # Replace with the actual VAE model for modality 3

# Load the trained weights for the VAE models
checkpoint_modality1 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint/vae_0_epoch_50.pth', map_location=torch.device('cpu'))
checkpoint_modality2 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint/vae_1_epoch_50.pth', map_location=torch.device('cpu'))
checkpoint_modality3 = torch.load('C:/Users/emend/3A_new/3A_new/Projet 3A/repo_final/cma/checkpoint/vae_2_epoch_50.pth', map_location=torch.device('cpu'))

vae_model_modality1.load_state_dict(checkpoint_modality1)
vae_model_modality2.load_state_dict(checkpoint_modality2)
vae_model_modality3.load_state_dict(checkpoint_modality3)

vae_model_modality1.eval()
vae_model_modality2.eval()
vae_model_modality3.eval()

# Fixer la graine pour Torch (pour la reproductibilité des résultats avec PyTorch)
torch.manual_seed(42)

##### MODALITY 1 #####

# Generate samples in modality 1
# num_samples = 10
num_samples = 50
latent_dim_modality1 = vae_model_modality1.encoder.z_mean.out_features

with torch.no_grad():
    random_latents_modality1 = torch.randn(num_samples, latent_dim_modality1)
    generated_samples_modality1_1 = vae_model_modality1.decoder(random_latents_modality1)

# Decode the samples into modality 2
with torch.no_grad():
    decoded_samples_modality2_1 = vae_model_modality2.decoder(random_latents_modality1)

# Decode the samples into modality 3
with torch.no_grad():
    decoded_samples_modality3_1 = vae_model_modality3.decoder(random_latents_modality1)    

# Print or use the generated and decoded samples as needed
# print("Generated samples in Modality 1:")
# print(generated_samples_modality1_1)

# print("\nDecoded samples in Modality 2:")
# print(decoded_samples_modality2_1)

# print("\nDecoded samples in Modality 3:")
# print(decoded_samples_modality3_1)

print("size modality 1 gene 1 : ", generated_samples_modality1_1.shape)
print("size modality 2 gene 1 : ", decoded_samples_modality2_1.shape)
print("size modality 3 gene 1 : ", decoded_samples_modality3_1.shape)

import matplotlib.pyplot as plt

# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality1 = generated_samples_modality1_1.numpy()
decoded_samples_modality2 = decoded_samples_modality2_1.numpy()
decoded_samples_modality3 = decoded_samples_modality3_1.numpy()

# Plot the samples
# plt.scatter(generated_samples_modality1_1[:, 0], generated_samples_modality1_1[:, 1], label='Generated Modality 1')
# plt.scatter(decoded_samples_modality2_1[:, 0], decoded_samples_modality2_1[:, 1], label='Decoded Modality 2')
# plt.scatter(decoded_samples_modality3_1[:, 0], decoded_samples_modality3_1[:, 1], label='Decoded Modality 3')

# plt.legend()
# plt.title('Generated and Decoded Samples')
# plt.show()



##### MODALITY 2 #####

# Generate samples in modality 2
# num_samples = 10
num_samples = 50
latent_dim_modality2 = vae_model_modality2.encoder.z_mean.out_features

with torch.no_grad():
    random_latents_modality2 = torch.randn(num_samples, latent_dim_modality2)
    generated_samples_modality2_2 = vae_model_modality2.decoder(random_latents_modality2)

# Decode the samples into modality 2
with torch.no_grad():
    decoded_samples_modality1_2 = vae_model_modality1.decoder(random_latents_modality2)
    
# Decode the samples into modality 3
with torch.no_grad():
    decoded_samples_modality3_2 = vae_model_modality3.decoder(random_latents_modality2)    

# Print or use the generated and decoded samples as needed
# print("Generated samples in Modality 2:")
# print(generated_samples_modality2_2)

# print("\nDecoded samples in Modality 1:")
# print(decoded_samples_modality1_2)

# print("\nDecoded samples in Modality 3:")
# print(decoded_samples_modality3_2)

print("size modality 1 gene 2 : ", decoded_samples_modality1_2.shape)
print("size modality 2 gene 2 : ", generated_samples_modality2_2.shape)
print("size modality 3 gene 2 : ", decoded_samples_modality3_2.shape)

import matplotlib.pyplot as plt

# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality2 = generated_samples_modality2_2.numpy()
decoded_samples_modality1 = decoded_samples_modality1_2.numpy()
decoded_samples_modality3 = decoded_samples_modality3_2.numpy()

# Plot the samples
# plt.scatter(generated_samples_modality2_2[:, 0], generated_samples_modality2_2[:, 1], label='Generated Modality 2')
# plt.scatter(decoded_samples_modality1_2[:, 0], decoded_samples_modality1_2[:, 1], label='Decoded Modality 1')
# plt.scatter(decoded_samples_modality3_2[:, 0], decoded_samples_modality3_2[:, 1], label='Decoded Modality 3')

# plt.legend()
# plt.title('Generated and Decoded Samples')
# plt.show()

##### MODALITY 3 #####

# Generate samples in modality 2
# num_samples = 10
num_samples = 50
latent_dim_modality3 = vae_model_modality3.encoder.z_mean.out_features

with torch.no_grad():
    random_latents_modality3 = torch.randn(num_samples, latent_dim_modality3)
    generated_samples_modality3_3 = vae_model_modality3.decoder(random_latents_modality3)

# Decode the samples into modality 2
with torch.no_grad():
    decoded_samples_modality1_3 = vae_model_modality1.decoder(random_latents_modality3)
    
# Decode the samples into modality 3
with torch.no_grad():
    decoded_samples_modality2_3 = vae_model_modality2.decoder(random_latents_modality3)    

# Print or use the generated and decoded samples as needed
# print("Generated samples in Modality 3:")
# print(generated_samples_modality3_3)

# print("\nDecoded samples in Modality 1:")
# print(decoded_samples_modality1_3)

# print("\nDecoded samples in Modality 2:")
# print(decoded_samples_modality2_3)


print("size modality 1 gene 3 : ", decoded_samples_modality1_3.shape)
print("size modality 2 gene 3 : ", decoded_samples_modality2_3.shape)
print("size modality 3 gene 3 : ", generated_samples_modality3_3.shape)

# Assuming your data is 2D, modify as needed for your actual data dimensions
generated_samples_modality2 = generated_samples_modality3_3.numpy()
decoded_samples_modality1 = decoded_samples_modality1_3.numpy()
decoded_samples_modality3 = decoded_samples_modality2_3.numpy()

# Plot the samples
# plt.scatter(generated_samples_modality3_3[:, 0], generated_samples_modality3_3[:, 1], label='Generated Modality 3')
# plt.scatter(decoded_samples_modality1_3[:, 0], decoded_samples_modality1_3[:, 1], label='Decoded Modality 1')
# plt.scatter(decoded_samples_modality2_3[:, 0], decoded_samples_modality2_3[:, 1], label='Decoded Modality 2')

# plt.legend()
# plt.title('Generated and Decoded Samples')
# plt.show()

########

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Remplacez ces matrices par vos propres données

# Liste des matrices
outputs = [generated_samples_modality1_1, decoded_samples_modality1_2,decoded_samples_modality1_3, generated_samples_modality2_2, decoded_samples_modality2_1, decoded_samples_modality2_3, generated_samples_modality3_3, decoded_samples_modality3_1, decoded_samples_modality3_2]

modalities = ['Modalité 1', 'Modalité 2', 'Modalité 3']

# Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, perplexity=20, n_iter = 1000, random_state = 42)

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


plt.figure(figsize=(15, 5))
for i in range(len(outputs)):
    reduced_data = tsne.fit_transform(outputs[i])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'{modalities[i//3]} depuis {modalities[i%3]}', marker='o')
plt.title('Génération de Modalités 1, 2 3 depuis Modalités 1, 2, 3')
plt.legend()

plt.tight_layout()
plt.show()

# # Génération de la modalité 1 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 1)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'{modalities[i]} depuis Modalité 1')
# plt.title('Génération des Modalités depuis Modalité 1')
# plt.legend()

# # Génération de la modalité 2 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 2)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i+3])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'{modalities[i]} depuis Modalité 2', marker='o')
# plt.title('Génération des Modalités depuis Modalité 2')
# plt.legend()

# # Génération de la modalité 3 à partir des modalités 1, 2, 3
# plt.subplot(1, 3, 3)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i+6])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'{modalities[i]} depuis Modalité 3', marker='o')
# plt.title('Génération des Modalités depuis Modalité 3')
# plt.legend()

# plt.tight_layout()
# plt.show()


# # Affichage des résultats
# plt.figure(figsize=(18, 6))  # Modifiez la taille ici

# # Génération de Modalité 1, 2, 3 à partir de Modalité 1
# plt.subplot(1, 3, 1)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1} depuis Modalité 1')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 1')
# plt.legend()

# # Génération de Modalité 1, 2, 3 à partir de Modalité 2
# plt.subplot(1, 3, 2)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i + 3])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1} depuis Modalité 2')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 2')
# plt.legend()

# # Génération de Modalité 1, 2, 3 à partir de Modalité 3
# plt.subplot(1, 3, 3)
# for i in range(3):
#     reduced_data = tsne.fit_transform(outputs[i + 6])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1} depuis Modalité 3')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 3')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Convertir la liste d'outputs en un tableau NumPy
# outputs_array = np.array(outputs)

# # Affichage des résultats
# plt.figure(figsize=(18, 6))

# # Appliquer t-SNE et afficher pour Modalité 1
# plt.subplot(1, 3, 1)
# for i in range(3):
#     start_index = i * 3
#     end_index = start_index + 10
#     reduced_data = tsne.fit_transform(outputs_array[start_index:end_index])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1}')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 1')
# plt.legend()

# # Appliquer t-SNE et afficher pour Modalité 2
# plt.subplot(1, 3, 2)
# for i in range(3):
#     start_index = i * 3 + 1
#     end_index = start_index + 10
#     reduced_data = tsne.fit_transform(outputs_array[start_index:end_index])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1}')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 2')
# plt.legend()

# # Appliquer t-SNE et afficher pour Modalité 3
# plt.subplot(1, 3, 3)
# for i in range(3):
#     start_index = i * 3 + 2
#     end_index = start_index + 10
#     reduced_data = tsne.fit_transform(outputs_array[start_index:end_index])
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Modalité {i + 1}')
# plt.title('Génération de Modalité 1, 2, 3 depuis Modalité 3')
# plt.legend()

# plt.tight_layout()
# plt.show()


# from scipy.stats import pearsonr
# from sklearn.preprocessing import StandardScaler

# outputs = [generated_samples_modality1_1.numpy(), decoded_samples_modality1_2.numpy(), decoded_samples_modality1_3.numpy(), 
#            generated_samples_modality2_2.numpy(), decoded_samples_modality2_1.numpy(), decoded_samples_modality2_3.numpy(), 
#            generated_samples_modality3_3.numpy(), decoded_samples_modality3_1.numpy(), decoded_samples_modality3_2.numpy()]
# modalities = ['Modalité 1', 'Modalité 2', 'Modalité 3']


# Réduction de dimension avec t-SNE
# tsne = TSNE(n_components=2, perplexity=8)

# # Normaliser les données
# scaled_outputs = [StandardScaler().fit_transform(data) for data in outputs]

# # Calculer les matrices de similarité entre les paires de modalités
# similarity_matrices = []

# for i in range(3):
#     start_index = i * 3
#     end_index = start_index + 3
#     similarity_matrix = np.zeros((3, 3))
    
#     for j in range(3):
#         modalite_start = j * 3
#         modalite_end = modalite_start + 1
#         group_data = scaled_outputs[start_index:end_index][modalite_start:modalite_end]
        
#         if group_data:
#             print(f"Shape of group_data before t-SNE: {np.vstack(group_data).shape}")
#             reduced_data = tsne.fit_transform(np.vstack(group_data))
#             print(f"Shape of reduced_data: {reduced_data.shape}")
#             similarity, _ = pearsonr(reduced_data[:, 0], reduced_data[:, 1])
#             similarity_matrix[i, j] = similarity

#     similarity_matrices.append(similarity_matrix)

# # Afficher les matrices de similarité
# for i, modality in enumerate(modalities):
#     print(f"\nSimilarité entre les paires de modalités pour {modality}:")
#     print(similarity_matrices[i])