import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


root_dir = os.path.join(os.path.dirname(__file__), '../raw') 

class IntersimDataset(Dataset):
    def __init__(self, data, labels):
        self.df = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx], self.labels[idx]


def get_paired_data(suffix, train_or_test):
        expr_X = pd.read_csv(root_dir + '/paired/' + suffix + '/' + train_or_test + '_gene_expression_' + suffix + '_paired.txt', sep='\t')
        methyl_X = pd.read_csv(root_dir + '/paired/' + suffix + '/' + train_or_test + '_dna_methylation_' + suffix + '_paired.txt', sep='\t')
        protein_X = pd.read_csv(root_dir + '/paired/' + suffix + '/' + train_or_test + '_protein_expression_' + suffix + '_paired.txt', sep='\t')
        y_common = pd.read_csv(root_dir + '/paired/' + suffix + '/' + train_or_test + '_labels_' + suffix + '_paired.txt', sep='\t')

        return (expr_X, y_common), (methyl_X, y_common), (protein_X, y_common)

def get_unpaired_data(suffix):
    expr_X = pd.read_csv(root_dir + '/unpaired/expression_unpaired_' + suffix + '.txt', sep='\t')
    expr_y = pd.read_csv(root_dir + '/unpaired/clusters_expression_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    methyl_X = pd.read_csv(root_dir + '/unpaired/methylation_unpaired_' + suffix + '.txt', sep='\t')
    methyl_y = pd.read_csv(root_dir + '/unpaired/clusters_methylation_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    protein_X = pd.read_csv(root_dir + '/unpaired/protein_unpaired_' + suffix + '.txt', sep='\t')
    protein_y = pd.read_csv(root_dir + '/unpaired/clusters_protein_unpaired_' + suffix + '.txt', sep='\t', index_col=0)

    return (expr_X, expr_y), (methyl_X, methyl_y), (protein_X, protein_y)


def concat_datasets(root_dir = root_dir, suffix = '3_clusters', type='paired'):
    # Noms des colonnes

    data_types = ['clusters', 'expression', 'methylation', 'protein']
    
    for data_type in data_types:
        column_names_written = False
        with open(root_dir + f'/paired/{data_type}_all_{suffix}.txt', 'w') as f_out:
            for i in [5, 10, 15]:
                with open(root_dir + f'/paired/{data_type}_{i}_{suffix}.txt') as f_in:
                    if not column_names_written:
                        f_out.write(f_in.readline())  # Écriture des noms de colonnes une seule fois
                        column_names_written = True
                    else:
                        f_in.readline()  # Passer les noms de colonnes dans les itérations suivantes
                    f_out.write(f_in.read())  # Écriture des données
    
    return "Concatenated paired datasets : Done"


def generate_datasets(suffix='3_clusters', type='paired', train=True, test=False):
    if type == 'paired':
        expr_train, methyl_train, protein_train = get_paired_data(suffix, train_or_test = 'train')
        print('Loading train paired dataset')
        expr_test, methyl_test, protein_test = get_paired_data(suffix, train_or_test = 'test')
        print('Loading test paired dataset')
    else:
        raise Exception('Invalid dataset type')

    expr_train_X, expr_train_y = expr_train
    expr_train_y = expr_train_y.iloc[:, 1].astype(int)
    methyl_train_X, methyl_train_y = methyl_train
    methyl_train_y = methyl_train_y.iloc[:, 1].astype(int)
    protein_train_X, protein_train_y = protein_train
    protein_train_y = protein_train_y.iloc[:, 1].astype(int)

    expr_test_X, expr_test_y = expr_test
    expr_test_y = expr_test_y.iloc[:, 1].astype(int)
    methyl_test_X, methyl_test_y = methyl_test
    methyl_test_y = methyl_test_y.iloc[:, 1].astype(int)
    protein_test_X, protein_test_y = protein_test
    protein_test_y = protein_test_y.iloc[:, 1].astype(int)

    # expr_train_X, expr_test_X, expr_train_y, expr_test_y = train_test_split(expr_X, expr_y, test_size=0.2, random_state=53)
    # methyl_train_X, methyl_test_X, methyl_train_y, methyl_test_y = train_test_split(methyl_X, methyl_y, test_size=0.2, random_state=53)
    # protein_train_X, protein_test_X, protein_train_y, protein_test_y = train_test_split(protein_X, protein_y, test_size=0.2, random_state=53)

    # Normaliser les données d'entraînement et de test
    import pickle
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    expr_train_X = scaler.fit_transform(expr_train_X)
    expr_test_X = scaler.transform(expr_test_X)

    methyl_train_X = scaler.fit_transform(methyl_train_X)
    methyl_test_X = scaler.transform(methyl_test_X)

    protein_train_X = scaler.fit_transform(protein_train_X)
    protein_test_X = scaler.transform(protein_test_X)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        print('Scaler saved')

    exp_train_dataset = IntersimDataset(expr_train_X, expr_train_y)
    exp_test_dataset = IntersimDataset(expr_test_X, expr_test_y)

    methyl_train_dataset = IntersimDataset(methyl_train_X, methyl_train_y)
    methyl_test_dataset = IntersimDataset(methyl_test_X, methyl_test_y)

    protein_train_dataset = IntersimDataset(protein_train_X, protein_train_y)
    protein_test_dataset = IntersimDataset(protein_test_X, protein_test_y)

    datasets = list()
    if train:
        datasets.extend([exp_train_dataset, methyl_train_dataset, protein_train_dataset])
    if test:
        datasets.extend([exp_test_dataset, methyl_test_dataset, protein_test_dataset])
    
    return datasets

