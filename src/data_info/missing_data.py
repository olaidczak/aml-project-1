import numpy as np
import pandas as pd
from scipy.special import expit

# Missing data generators (Y = -1 if there is missing value)
def apply_missingness(y, probabilities):
    S = np.random.binomial(1, probabilities)
    return np.where(S == 1, -1, y)

def generate_mcar(X, y, c=0.3):
    # Missing Completely at Random
    probabilities = np.full(len(y), c)
    return pd.Series(apply_missingness(y, probabilities), name='Y_obs')

def generate_mar1(X, y):
    # Missing at Random (depends on 1 feature with max variance)
    feature = X.iloc[:, X.var().argmax()].values
    feature_norm = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
    probabilities = expit(feature_norm)
    return pd.Series(apply_missingness(y, probabilities), name='Y_obs')

def generate_mar2(X, y):
    # Missing at Random (depends on all features)
    weights = np.random.randn(X.shape[1])
    linear_comb = np.dot(X, weights)
    linear_comb_norm = (linear_comb - np.mean(linear_comb)) / (np.std(linear_comb) + 1e-8)
    probabilities = expit(linear_comb_norm - 0.5)
    return pd.Series(apply_missingness(y, probabilities), name='Y_obs')
'''eksperyment 1
def generate_mnar(X, y, gamma=10.0, n1=400):
    """
    Generuje braki danych typu self-masked MNAR zgodnie z metodą z artykułu 
    (Sportisse et al., 2023, sekcja 5.1).
    
    Liczba etykietowanych przykładów w klasie k jest określona wzorem:
    n_k = n1 * gamma ** (-(k-1) / (K-1))
    """
    y_true = np.asarray(y)
    classes = np.unique(y_true)
    K = len(classes)
    
    # Inicjalizacja: -1 oznacza brak danych (unlabeled)
    y_obs = np.full(y_true.shape, -1, dtype=float)
    
    for i, k_val in enumerate(classes):
        # Indeksy wszystkich przykładów należących do danej klasy
        indices_k = np.where(y_true == k_val)[0]
        
        # Obliczanie docelowej liczby etykietowanych próbek dla tej klasy (n_k)
        # Wzór z artykułu: n_k = n1 * gamma^(-(k-1)/(K-1)) 
        n_k = int(n1 * (gamma ** (-(i) / (K - 1))))
        
        # Zabezpieczenie, by nie wybrać więcej próbek niż jest dostępnych
        n_k = min(n_k, len(indices_k))
        
        if n_k > 0:
            # Losowo wybieramy n_k przykładów, które pozostaną widoczne (S=0)
            observed_indices = np.random.choice(indices_k, n_k, replace=False)
            y_obs[observed_indices] = k_val
            
    return pd.Series(y_obs, name='Y_obs')
    '''

def generate_mnar(X, y, ratio=0.2, gamma=3.0, min_labels=5):
    """
    ratio: jaki ułamek klasy 0 zostawiamy (np. 0.2 = 20%).
    gamma: jak silnie ograniczamy etykiety dla kolejnych klas.
    min_labels: absolutne minimum etykiet na klasę, by uniknąć F1=0.
    """
    y_true = np.asarray(y)
    classes = np.unique(y_true)
    y_obs = np.full(y_true.shape, -1, dtype=float)
    
    for i, k_val in enumerate(classes):
        indices_k = np.where(y_true == k_val)[0]
        
        # Obliczamy n_k jako procent wielkości klasy
        target_ratio = ratio / (gamma ** i)
        n_k = int(len(indices_k) * target_ratio)
        
        # Gwarancja minimalnej liczby etykiet dla stabilności numerycznej
        n_k = max(min(n_k, len(indices_k)), min_labels)
        
        if n_k > 0:
            observed_indices = np.random.choice(indices_k, n_k, replace=False)
            y_obs[observed_indices] = k_val
            
    return pd.Series(y_obs, name='Y_obs')