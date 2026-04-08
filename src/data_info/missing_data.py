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

def generate_mnar(X, y):
    # Missing Not at Random (depends on features and unobserved true Y)
    weights = np.random.randn(X.shape[1])
    linear_comb = np.dot(X, weights)
    linear_comb_norm = (linear_comb - np.mean(linear_comb)) / (np.std(linear_comb) + 1e-8)
    y_influence = 2.0 * (y.values - 0.5)
    probabilities = expit(linear_comb_norm + y_influence)
    return pd.Series(apply_missingness(y, probabilities), name='Y_obs')