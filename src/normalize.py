from sklearn.preprocessing import StandardScaler
import os 
import pandas as pd 

# Concaténation des données et normalisatioin 
def normalize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

paired_dir = '/Users/meliya/Desktop/Projet 3A/cma/data/paired'

# Liste pour stocker les données concaténées
all_expression = []
all_methylation = []
all_protein = []

for filename in os.listdir(paired_dir):
    if filename.startswith("expression"):
        expression_data = pd.read_csv(os.path.join(paired_dir, filename), sep='\t')
        all_expression.append(expression_data)
    elif filename.startswith("methylation"):
        methylation_data = pd.read_csv(os.path.join(paired_dir, filename), sep='\t')
        all_methylation.append(methylation_data)
    elif filename.startswith("protein"):
        protein_data = pd.read_csv(os.path.join(paired_dir, filename), sep='\t')
        all_protein.append(protein_data)

concatenated_expression = pd.concat(all_expression, axis=1)
concatenated_methylation = pd.concat(all_methylation, axis=1)
concatenated_protein = pd.concat(all_protein, axis=1)

normalized_expression = normalize_data(concatenated_expression)
normalized_methylation = normalize_data(concatenated_methylation)
normalized_protein = normalize_data(concatenated_protein)

normalized_expression_df = pd.DataFrame(normalized_expression, columns=concatenated_expression.columns)
normalized_methylation = pd.DataFrame(normalized_methylation, columns=concatenated_methylation.columns)
normalized_protein = pd.DataFrame(normalized_protein, columns=concatenated_protein.columns)

output_dir = '/Users/meliya/Desktop/Projet 3A/cma/data/paired/normalized/'

os.makedirs(output_dir, exist_ok=True)

normalized_expression_df.to_csv(os.path.join(output_dir, 'normalized_expression.txt'), sep='\t', index=False)
normalized_methylation.to_csv(os.path.join(output_dir, 'normalized_methylation.txt'), sep='\t', index=False)
normalized_protein.to_csv(os.path.join(output_dir, 'normalized_protein.txt'), sep='\t', index=False)



#Normalisation de chaque fichier 

for filename in os.listdir(paired_dir):
    if filename.startswith("expression") or filename.startswith("methylation") or filename.startswith("protein"):

        data = pd.read_csv(os.path.join(paired_dir, filename), sep='\t')
        
        normalized_data = normalize_data(data)
        
        normalized_filename = "normalized_" + filename
        
        normalized_data_df = pd.DataFrame(normalized_data, columns=data.columns)
        
        normalized_data_df.to_csv(os.path.join(output_dir, normalized_filename), sep='\t', index=False)


