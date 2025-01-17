import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Connect to the database and load data
def load_data(database, query):
    conn = sqlite3.connect(database)
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Load main hardware data
data = load_data("hardware.db", "SELECT * FROM hardware;")

# Load component prices from respective tables
def load_component_prices(database):
    conn = sqlite3.connect(database)
    cpu_prices = pd.read_sql_query("SELECT id, price AS CPU_Price FROM Cpu;", conn)
    gpu_prices = pd.read_sql_query("SELECT id, price AS GPU_Price FROM Gpu;", conn)
    ram_prices = pd.read_sql_query("SELECT id, capacity, price AS RAM_Price FROM Ram;", conn)
    mbd_prices = pd.read_sql_query("SELECT id, upgrade, price AS MBD_Price FROM Mbd;", conn)
    hdd_prices = pd.read_sql_query("SELECT id, capacity AS hdd_capacity, price AS HDD_Price FROM Hdd;", conn)
    ssd_prices = pd.read_sql_query("SELECT id, capacity AS ssd_capacity, price AS SSD_Price FROM Ssd;", conn)
    conn.close()
    return cpu_prices, gpu_prices, ram_prices, mbd_prices, hdd_prices, ssd_prices

# Call the function to load data
cpu_prices, gpu_prices, ram_prices, mbd_prices, hdd_prices, ssd_prices = load_component_prices("hardware.db")

# Merge component prices into the main data with explicit suffixes
data = data.merge(cpu_prices, left_on='CPU', right_on='id', how='left', suffixes=('', '_CPU'))
data = data.merge(gpu_prices, left_on='GPU', right_on='id', how='left', suffixes=('', '_GPU'))
data = data.merge(ram_prices, left_on='RAM', right_on='id', how='left', suffixes=('', '_RAM'))
data = data.merge(mbd_prices, left_on='MBD', right_on='id', how='left', suffixes=('', '_MBD'))
data = data.merge(ssd_prices, left_on='SSD', right_on='id', how='left', suffixes=('', '_SSD'))
data = data.merge(hdd_prices, left_on='HDD', right_on='id', how='left', suffixes=('', '_HDD'))

# Drop unnecessary columns if they exist
columns_to_drop = ['id_CPU', 'id_GPU', 'id_RAM', 'id_MBD','id_HDD','id_SSD']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Convert scores to numeric where necessary
data['gaming_score'] = data['gaming_score'].astype(float)
data['desk_score'] = data['desk_score'].astype(float)
data['work_score'] = data['work_score'].astype(float)
data['SSD_Price'] = data['SSD_Price'].fillna(0)
data['SSD_Price'] = data['HDD_Price'].fillna(0)
data['ssd_capacity'] = data['ssd_capacity'].fillna(0)
data['hdd_capacity'] = data['hdd_capacity'].fillna(0)
data['capacite_(memoire)'] = data['ssd_capacity'].astype(float)+data['hdd_capacity'].astype(float)
data['amelioration'] = data['upgrade'].astype(float)


# Calculate total price
data['Price'] = data['CPU_Price'] + data['GPU_Price'] + data['RAM_Price'] + data['MBD_Price'] + data['SSD_Price'] + data['HDD_Price']

# Extract brand preferences
data['CPU_Brand'] = data['CPU'].str[0].map({'I': 'Intel', 'A': 'AMD'})
data['GPU_Brand'] = data['GPU'].str[0].map({'N': 'Nvidia', 'A': 'AMD'})

def thresh_criteria(criterion, sensitivity):
    if criterion == 'Price' :
        if sensitivity == "1":
            return {'q': 50, 'p': 150, 'v': 250}
        elif sensitivity == "2":
            return {'q': 25, 'p': 100, 'v': 200}
        elif sensitivity == "3":
            return {'q': 10, 'p': 50, 'v': 100}
        else:
            print("Entrée invalide, par défaut '2'.")
            return {'q': 25, 'p': 100, 'v': 200}
    elif criterion == 'amelioration' :
        if sensitivity == "1":
            return {'q': 1, 'p': 3, 'v': 5}
        elif sensitivity == "2":
            return {'q': 0, 'p': 2, 'v': 3}
        elif sensitivity == "3":
            return {'q': 0, 'p': 1, 'v': 2}
        else:
            print("Entrée invalide, par défaut '2'.")
            return {'q': 0, 'p': 2, 'v': 4}        
    else :
        if sensitivity == "1":
            return {'q': 10, 'p': 30, 'v': 50}
        elif sensitivity == "2":
            return {'q': 5, 'p': 20, 'v': 40}
        elif sensitivity == "3":
            return {'q': 2, 'p': 10, 'v': 20}
        else:
            print("Entrée invalide, par défaut '2'.")
            return {'q': 5, 'p': 20, 'v': 40}   



# Add RAM capacity as a separate criterion
def infer_user_preferences(criteria):
    print("\nNous allons poser des questions pour définir vos préférences.")
    weights = {}
    thresholds = {}

    for criterion in criteria:
        print(f"\nQuelle est l'importance de {criterion.replace('_', ' ')} pour vous ?")
        print("1: Peu important, 2: Assez important, 3: Très important")
        importance = input("Entrez votre choix (1, 2 ou 3) : ").strip()

        if importance == "1":
            weights[criterion] = 0.1
        elif importance == "2":
            weights[criterion] = 0.3
        elif importance == "3":
            weights[criterion] = 0.6
        else:
            print("Entrée invalide, par défaut '2'.")
            weights[criterion] = 0.3

        print(f"\nQuelle est votre sensibilité aux différences de {criterion.replace('_', ' ')} ?")
        print("1: Faible, 2: Moyenne, 3: Forte")
        sensitivity = input("Entrez votre choix (1, 2 ou 3) : ").strip()
        thresholds[criterion] = thresh_criteria(criterion, sensitivity)

    max_budget = float(input("\nQuel est votre budget maximum pour le matériel ? : "))

    print("\nQuelle marque de CPU préférez-vous ?")
    print("1: Intel, 2: AMD, 3: Aucune")
    cpu_preference_choice = input("Entrez votre choix (1, 2 ou 3) : ").strip()
    cpu_preference = "Intel" if cpu_preference_choice == "1" else "AMD" if cpu_preference_choice == "2" else None

    print("\nQuelle marque de GPU préférez-vous ?")
    print("1: Nvidia, 2: AMD, 3: Aucune")
    gpu_preference_choice = input("Entrez votre choix (1, 2 ou 3) : ").strip()
    gpu_preference = "Nvidia" if gpu_preference_choice == "1" else "AMD" if gpu_preference_choice == "2" else None

    return weights, thresholds, max_budget, cpu_preference, gpu_preference



criteria = ['gaming_score', 'work_score', 'desk_score', 'capacite_(memoire)', 'amelioration', 'Price', 'CPU_Brand', 'GPU_Brand']
weights, thresholds, max_budget, cpu_preference, gpu_preference = infer_user_preferences(criteria)
# Filter data based on budget
data = data[data['Price'] <= max_budget]
def concordance(a, b, criteria, thresholds):
    c = []
    for criterion in criteria:
        if criterion == 'CPU_Brand':
            # Si l'utilisateur préfère une marque spécifique de CPU
            if a['CPU_Brand'] == cpu_preference and b['CPU_Brand'] != cpu_preference:
                c.append(1)
            elif b['CPU_Brand'] == cpu_preference and a['CPU_Brand'] != cpu_preference:
                c.append(0)
            else:
                c.append(0.5)  # Indifférence si les deux ont des marques non préférées ou identiques

        elif criterion == 'GPU_Brand':
            # Si l'utilisateur préfère une marque spécifique de GPU
            if a['GPU_Brand'] == gpu_preference and b['GPU_Brand'] != gpu_preference:
                c.append(1)
            elif b['GPU_Brand'] == gpu_preference and a['GPU_Brand'] != gpu_preference:
                c.append(0)
            else:
                c.append(0.5)  # Indifférence

        else:
            # Critères classiques
            diff = a[criterion] - b[criterion]
            if diff >= -thresholds[criterion]['q']:
                c.append(1)
            elif diff <= -thresholds[criterion]['p']:
                c.append(0)
            else:
                c.append((thresholds[criterion]['p'] + diff) / (thresholds[criterion]['p'] - thresholds[criterion]['q']))
    return c


# Discordance index calculation
def discordance(a, b, criteria, thresholds):
    d = []
    for criterion in criteria:
        if criterion in ['CPU_Brand', 'GPU_Brand']:
            # Skip non-numeric criteria for discordance calculations
            continue
        diff = a[criterion] - b[criterion]
        if diff >= -thresholds[criterion]['v']:
            d.append(0)
        elif diff <= -thresholds[criterion]['p']:
            d.append(1)
        else:
            d.append((thresholds[criterion]['v'] + diff) / (thresholds[criterion]['v'] - thresholds[criterion]['p']))
    return d


# Global concordance index
def global_concordance(a, b, criteria, thresholds, weights):
    c = concordance(a, b, criteria, thresholds)
    return sum(w * ci for w, ci in zip(weights.values(), c))

# Credibility index calculation
def credibility(a, b, criteria, thresholds, weights):
    c_global = global_concordance(a, b, criteria, thresholds, weights)
    d = discordance(a, b, criteria, thresholds)
    credible = c_global
    for i, di in enumerate(d):
        if di > c_global:
            credible *= (1 - di) / (1 - c_global)
    return credible

# Build pairwise relations
def build_relation(data, criteria, thresholds, weights, credibility_threshold):
    n = len(data)
    relation_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cred = credibility(data.iloc[i], data.iloc[j], criteria, thresholds, weights)
                relation_matrix[i, j] = cred if cred >= credibility_threshold else 0
    return relation_matrix

credibility_threshold = 0.6
relation_matrix = build_relation(data, criteria, thresholds, weights, credibility_threshold)

def find_most_efficient_builds(relation_matrix):
    n = relation_matrix.shape[0]
    dominance_counts = relation_matrix.sum(axis=1)
    sorted_indices = np.argsort(-dominance_counts)  # Sort by dominance in descending order
    return sorted_indices, dominance_counts

sorted_indices, dominance_counts = find_most_efficient_builds(relation_matrix)

# Print sorted configurations
print("Configuration classées par dominance :")
for rank, idx in enumerate(sorted_indices, start=1): 
    print(f"Rang {rank}: Prix ${data.iloc[idx]['Price']:.2f}, CPU: {data.iloc[idx]['CPU']}, GPU: {data.iloc[idx]['GPU']} avec un score de dominance {dominance_counts[idx]:.2f}, Capacité : {data.iloc[idx]['capacite_(memoire)']} GB")


