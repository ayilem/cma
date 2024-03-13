import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models_mel import MultimodalAE
from src.dataset import generate_datasets
from src.functions import Log
from src.config import config as default_config
from sklearn.preprocessing import StandardScaler
import sys 



script_dir = os.path.dirname(__file__)


#-------------------------------------------------



train_datasets = generate_datasets(suffix='5_diff', type='paired', train=True, test=False)
test_datasets = generate_datasets(suffix='5_diff', type='paired', train=False, test=True)


train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets]
test_loaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets]


n_inputs1 = train_datasets[0][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 1
n_inputs2 = train_datasets[1][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 2
n_outputs = train_datasets[2][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 3
                              
print(n_inputs1, n_inputs2, n_outputs)
latent_dims = 20   
n_hiddens = 256 

model = MultimodalAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)

critere = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())



# Entraînement du modèle
num_epochs = 10  
for epoch in range(num_epochs):
    model.train()
    loss_sum = 0
    num_batches = 0  

    for (x1, _), (x2, _), (y, _) in zip(*train_loaders):
        

        # Forward pass
        outputs = model(x1, x2)
        loss = critere(outputs, y)
        loss_sum += loss.item() 
        num_batches += 1  

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = loss_sum / num_batches  
    print(f'Epoch {epoch+1}, Train Average Loss: {avg_loss}')


model.eval()
with torch.no_grad():
    total_loss = 0
    all_losses = []  
    for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
        predictions = model(x1, x2)
        loss = critere(predictions, y)
        total_loss += loss.item()
        all_losses.append(loss.item())  
    print(f'Test Average Loss: {sum(all_losses) / len(all_losses)}')
        
    




#---------------CROSS VALDIATION-------------------

from torch.utils.data import Subset
import torch
from sklearn.model_selection import KFold



n_splits = 10
kf = KFold(n_splits=n_splits)


datasets = generate_datasets(suffix='5_diff', type='paired', train=True, test=False)


datasets = [list(dataset) for dataset in datasets]


all_fold_losses = []

for fold, (train_index, test_index) in enumerate(kf.split(datasets[0])):
    print(f'Fold {fold+1}')
    fold_losses = []

   
    train_datasets = [Subset(dataset, train_index) for dataset in datasets]
    test_datasets = [Subset(dataset, test_index) for dataset in datasets]

 
    train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets]
    test_loaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets]


    model = MultimodalAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)
    optimizer = torch.optim.Adam(model.parameters())

   
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for (x1, _), (x2, _), (y, _) in zip(*train_loaders):
            if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
                continue
            outputs = model(x1, x2)
            loss = critere(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item()) 

        avg_loss = sum(epoch_losses) / len(epoch_losses)  
        print(f'Epoch {epoch+1}, Train Average Loss: {avg_loss}')
        fold_losses.append(avg_loss)  
    all_fold_losses.append(fold_losses)  

print("All Fold Losses:")
for fold, fold_losses in enumerate(all_fold_losses):
    print(f'Fold {fold+1}:')
    for epoch, loss in enumerate(fold_losses):
        print(f'Epoch {epoch+1}, Average Loss: {loss}')
