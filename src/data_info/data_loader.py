# Data loaders (majority class - positive (1), rest - negative (0))
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

def load_openml_data(data_id):
    # We use as_frame=False to avoid the Sparse ARFF vs Pandas conflict
    # and parser='liac-arff' or 'auto' to handle the sparse format properly
    data = fetch_openml(data_id=data_id, as_frame=False, parser='auto')
    
    X = data.data
    y = data.target

    # 1. Handle Sparse Matrices (common in Gisette)
    # If X is a scipy sparse matrix (csr_matrix, etc.), convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # 2. Convert to DataFrame
    # fetch_openml with as_frame=False returns numpy arrays/scipy matrices
    # We use data.feature_names to keep the original column names if available
    feature_names = getattr(data, 'feature_names', [f"feature_{i}" for i in range(X.shape[1])])
    X = pd.DataFrame(X, columns=feature_names)

    # 3. Select only numeric columns
    X = X.select_dtypes(include=[np.number])

    # 4. Binary label conversion
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    unique_classes = y.unique()
    
    if len(unique_classes) == 2:
        # binary y
        positive_class = unique_classes[0]
        y_binary = (y == positive_class).astype(int)
    else:
        # multiclass y - cast to majority class
        majority_class = y.mode()[0]
        y_binary = (y == majority_class).astype(int)

    return X, y_binary


def load_uci_data(repo_id, binarize_median=False):
    dataset = fetch_ucirepo(id=repo_id)
    X = dataset.data.features.select_dtypes(include=[np.number])
    y = dataset.data.targets.iloc[:, 0]

    if binarize_median and pd.api.types.is_numeric_dtype(y):
        # continuous y (e.g. Communities and Crime)
        median_val = y.median()
        y_binary = (y > median_val).astype(int)
    else:
        unique_classes = y.unique()
        if len(unique_classes) == 2:
            # binary y
            positive_class = unique_classes[0]
            y_binary = (y == positive_class).astype(int)
        else:
            # multiclass y
            majority_class = y.mode()[0]
            y_binary = (y == majority_class).astype(int)

    return X, y_binary


def preprocess_data(X, threshold=0.9):
    # 1. Imputacja braków (zgodnie z Task 1)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2. Usuwanie kolinearnych zmiennych (zgodnie z Task 1)
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X_imputed.drop(columns=to_drop)

    # 3. SKALOWANIE - Kluczowe dla zbieżności FISTA
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_reduced), columns=X_reduced.columns)

    return X_scaled