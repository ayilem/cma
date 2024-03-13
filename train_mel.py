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

train_datasets = generate_datasets(suffix='5_diff', type='paired', train=True, test=False)
test_datasets = generate_datasets(suffix='5_diff', type='paired', train=False, test=True)


train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets]
test_loaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets]


n_inputs1 = train_datasets[0][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 1
n_inputs2 = train_datasets[1][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 2
n_outputs = train_datasets[2][0][0].shape[0]  # La taille du vecteur de caractéristiques pour la modalité 3
                              
latent_dims = 20   
n_hiddens = 256 

model = MultimodalVAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)

critere = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())



# Entraînement du modèle
num_epochs = 10  
for epoch in range(num_epochs):
    model.train()
    for (x1, _), (x2, _), (y, _) in zip(*train_loaders):

        if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
            continue

    
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
    all_losses = []  
    for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
        predictions = model(x1, x2)
        loss = critere(predictions, y)
        total_loss += loss.item()
        all_losses.append(loss.item())  
    print(f'Test Loss: {total_loss / len(test_loaders[0])}')
    print(f'All Losses: {all_losses}')
        
    



#---------------KFold 


# from sklearn.model_selection import KFold

# k_folds = 10
# num_runs = 10

# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# all_test_losses = []

# for run in range(num_runs):
#     print(f"Run {run+1}/{num_runs}")
#     test_losses = []  
    
#     for train_index, test_index in kf.split(train_datasets[0]):
#         train_datasets_fold = [train_datasets[i][train_index] for i in range(len(train_datasets))]
#         test_datasets_fold = [train_datasets[i][test_index] for i in range(len(train_datasets))]

#         train_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in train_datasets_fold]
#         test_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in test_datasets_fold]

        # for i, (train_data, test_data) in enumerate(zip(train_datasets_fold, test_datasets_fold)):
        #     print(f"Fold {i+1}")
        #     print("Train data shapes:")
        #     for data in train_data:
        #         print(data.shape)
        #     print("Test data shapes:")
        #     for data in test_data:
        #         print(data.shape)


        # for epoch in range(num_epochs):
        #     model.train()
        #     for (x1, _), (x2, _), (y, _) in zip(*train_loaders_fold):
        #         if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
        #             continue

        #         outputs = model(x1, x2)
        #         loss = critere(outputs, y)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             total_loss = 0
#             for (x1, _), (x2, _), (y, _) in zip(*test_loaders_fold):
#                 predictions = model(x1, x2)
#                 loss = critere(predictions, y)
#                 total_loss += loss.item()
#             test_loss = total_loss / len(test_loaders_fold[0])
#             test_losses.append(test_loss)

#     avg_test_loss = sum(test_losses) / len(test_losses)
#     all_test_losses.append(avg_test_loss)

# # Calculer la moyenne des performances du modèle sur tous les runs
# avg_test_loss_over_runs = sum(all_test_losses) / len(all_test_losses)
# print(f'Average Test Loss over {num_runs} runs: {avg_test_loss_over_runs}')






#-------------------------------------------------
                
# import argparse
# import os
# import torch
# from torch.utils.data import DataLoader
# from models_mel import MultimodalVAE
# from src.dataset import generate_datasets
# from src.functions import Log
# from src.config import config as default_config

# from sklearn.model_selection import KFold

# # K-fold cross-validation
# k_folds = 10
# num_runs = 10

# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# all_test_losses = []

# for run in range(num_runs):
#     print(f"Run {run+1}/{num_runs}")
#     test_losses = []

#     for train_index, test_index in kf.split(train_datasets):
#         train_datasets_fold = [train_datasets[i][train_index] for i in range(len(train_datasets))]
#         test_datasets_fold = [train_datasets[i][test_index] for i in range(len(train_datasets))]

#         tensor_train_datasets_fold = [torch.tensor(dataset[0], dtype=torch.float32) for dataset in train_datasets_fold]
#         tensor_test_datasets_fold = [torch.tensor(dataset[0], dtype=torch.float32) for dataset in test_datasets_fold]

#         train_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in tensor_test_datasets_fold]
#         print(train_loaders_fold)
#         test_loaders_fold = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in tensor_test_datasets_fold]

#         n_inputs1 = train_datasets_fold[0][0][0].shape[0]
#         n_inputs2 = train_datasets_fold[1][0][0].shape[0]
#         n_outputs = train_datasets_fold[2][0][0].shape[0]

#         n_hiddens = 256
#         latent_dims = 20

#         model = MultimodalVAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)
#         optimizer = torch.optim.Adam(model.parameters())

#         for epoch in range(num_epochs):
#             model.train()
#             for ((x1, _), (x2, _)), (y, _) in zip(*train_loaders_fold):
#                 if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
#                     continue

#                 outputs = model(x1, x2)
#                 loss = critere(outputs, y)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             total_loss = 0
#             for ((x1, _), (x2, _)), (y, _) in zip(*test_loaders_fold):
#                 predictions = model(x1, x2)
#                 loss = critere(predictions, y)
#                 total_loss += loss.item()

#             test_loss = total_loss / len(test_loaders_fold[0])
#             test_losses.append(test_loss)

#     avg_test_loss = sum(test_losses) / len(test_losses)
#     all_test_losses.append(avg_test_loss)

# avg_test_loss_over_runs = sum(all_test_losses) / len(all_test_losses)
# print(f'Average Test Loss over {num_runs} runs: {avg_test_loss_over_runs}')
    


#-------------------------------------------------
    
    import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import generate_datasets
from src.functions import Log
from src.config import config as default_config
from sklearn.model_selection import KFold
import numpy as np

num_folds = 5

train_datasets = generate_datasets(suffix='5_diff', type='paired', train=True, test=False)
test_datasets = generate_datasets(suffix='5_diff', type='paired', train=False, test=True)

n_inputs1 = train_datasets[0][0][0].shape[0]
n_inputs2 = train_datasets[1][0][0].shape[0]
n_outputs = train_datasets[2][0][0].shape[0]
latent_dims = 20
n_hiddens = 256
model = MultimodalVAE(n_inputs1=n_inputs1, n_inputs2=n_inputs2, latent_dims=latent_dims, n_hiddens=n_hiddens, n_outputs=n_outputs)
critere = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

kf = KFold(n_splits=num_folds)
all_losses = []
for fold, (train_index, test_index) in enumerate(kf.split(train_datasets[0])):
    print(f'Fold {fold + 1}')
    train_loaders = [DataLoader(dataset[train_index], batch_size=32, shuffle=True) for dataset in train_datasets]
    test_loaders = [DataLoader(dataset[test_index], batch_size=32, shuffle=False) for dataset in test_datasets]
    print (train_loaders)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for (x1, _), (x2, _), (y, _) in zip(*train_loaders):
            if x1.size(0) != 32 or x2.size(0) != 32 or y.size(0) != 32:
                continue
            outputs = model(x1, x2)
            loss = critere(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        all_losses = []
        for (x1, _), (x2, _), (y, _) in zip(*test_loaders):
            predictions = model(x1, x2)
            loss = critere(predictions, y)
            total_loss += loss.item()
            all_losses.append(loss.item())
        print(f'Test Loss: {total_loss / len(test_loaders[0])}')
        print(f'All Losses: {all_losses}')

print(f'Average Loss: {np.mean(all_losses)}')

