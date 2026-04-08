# Data loaders (majority class - positive (1), rest - negative (0))
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

def load_openml_data(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    X = data.data.select_dtypes(include=[np.number])
    y = data.target

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

# Preprocessing
def preprocess_data(X, threshold=0.9):
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Drop highly correlated features (>0.9)
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_final = X_imputed.drop(columns=to_drop)
    return X_final

