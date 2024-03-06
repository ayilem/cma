import argparse
import os
import torch
from torch.utils.data import DataLoader
from models_mel import MultimodalVAE
from src.dataset import generate_datasets
from src.functions import Log
from src.config import config as default_config


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
n_outputs = train_datasets[2][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 3
                              
# Obtenez les dimensions de l'espace latent et du réseau caché à partir de config.py
latent_dims = 20   
n_hiddens = 256 

# Création du modèle
model = MultimodalVAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)

# Choix de la fonction de perte et de l'optimiseur
critere = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())



# Entraînement du modèle
num_epochs = 10  # Définissez le nombre d'époques que vous voulez
for epoch in range(num_epochs):
    model.train()
    for (x1, _), (x2, _), (y, _) in zip(*train_loaders):

        if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
            continue

        # Forward pass
        # outputs = model(x1, x2)
        # reconstructions, mu1, logvar1, mu2, logvar2 = model(x1, x2)
        # # print("reconstructions:", reconstructions.size())
        # # print("taille de mu1:", mu1.size())
        # # print("logvar1:", logvar1.size())
        # # print("mu2:", mu2.size()) 
        # # print("logvar2:", logvar2.size())
        
        # # reconstructions, mu1, logvar1, mu2, logvar2 = model(x1, x2)
        # # loss = critere(reconstructions, y)
        # print(outputs.shape)
        # print(y.shape)
        # loss = critere(outputs, y)

        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()



       # Forward pass
        outputs = model(x1, x2)
        print(outputs.shape)
        print(y.shape)
        loss = critere(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
with torch.no_grad():
    total_loss = 0
    all_losses = []  # Liste pour stocker toutes les pertes
    for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
        predictions = model(x1, x2)
        loss = critere(predictions, y)
        total_loss += loss.item()
        all_losses.append(loss.item())  # Ajoutez la perte actuelle à la liste
    print(f'Test Loss: {total_loss / len(test_loaders[0])}')
    print(f'All Losses: {all_losses}')
        
    



#---------------KFold 


from sklearn.model_selection import KFold

# Définir le nombre de plis et de runs
k_folds = 10
num_runs = 10

# Initialiser la k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Liste pour stocker les performances du modèle sur chaque run
all_test_losses = []

# Boucle sur les runs
for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")
    test_losses = []  # Liste pour stocker les pertes du modèle sur chaque pli
    
    # Boucle sur les plis de la k-fold cross-validation
    for train_index, test_index in kf.split(train_datasets[0]):
        # Diviser les données en ensembles d'entraînement et de test pour ce pli
        train_datasets_fold = [train_datasets[i][train_index] for i in range(len(train_datasets))]
        test_datasets_fold = [train_datasets[i][test_index] for i in range(len(train_datasets))]

        # Création des chargeurs de données
        train_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets_fold]
        test_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets_fold]

        # Entraînement du modèle
        for epoch in range(num_epochs):
            model.train()
            for (x1, _), (x2, _), (y, _) in zip(*train_loaders_fold):
                if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
                    continue

                outputs = model(x1, x2)
                loss = critere(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Évaluation du modèle
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for (x1, _), (x2, _), (y, _) in zip(*test_loaders_fold):
                predictions = model(x1, x2)
                loss = critere(predictions, y)
                total_loss += loss.item()
            test_loss = total_loss / len(test_loaders_fold[0])
            test_losses.append(test_loss)

    # Calculer la moyenne des pertes du modèle sur les plis de ce run
    avg_test_loss = sum(test_losses) / len(test_losses)
    all_test_losses.append(avg_test_loss)

# Calculer la moyenne des performances du modèle sur tous les runs
avg_test_loss_over_runs = sum(all_test_losses) / len(all_test_losses)
print(f'Average Test Loss over {num_runs} runs: {avg_test_loss_over_runs}')






