import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_openml_data(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from OpenML.

    Args:
        data_id: OpenML dataset identifier.

    Returns:
        Tuple of (X, y_binary) where X is a DataFrame of numeric features
        and y_binary is a binary Series with values in {0, 1}.
    """
    data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    X = data.data
    y = data.target

    if hasattr(X, "toarray"):
        X = X.toarray()
    feature_names = getattr(
        data, "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
    )
    X = pd.DataFrame(X, columns=feature_names)
    X = X.select_dtypes(include=[np.number])

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


def preprocess_data(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Impute missing values, drop highly correlated features and standardise.

    Pipeline: mean imputation, removal of features whose pairwise absolute
    correlation exceeds threshold, and z-score standardisation.

    Args:
        X: Raw feature matrix.
        threshold: Correlation cutoff — features with absolute pairwise
            correlation above this value are dropped (default: 0.9).

    Returns:
        Preprocessed DataFrame.
    """
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X_imputed.drop(columns=to_drop)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_reduced), columns=X_reduced.columns)

    return X_scaled


def drop_corr_features(X, y, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

def scale_after_split(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train_scl = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_valid_scl = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)
    X_test_scl = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scl, X_valid_scl, X_test_scl


def preprocess_after_split(X_train, X_valid, X_test, threshold=0.9):
    """
    Wykonuje preprocesing zabezpieczając przed wyciekiem danych (Data Leakage).
    Fituje transformatory TYLKO na zbiorze treningowym.
    """
    # 1. Imputacja braków na podstawie średnich z TRAIN
    # imputer = SimpleImputer(strategy="mean")
    # X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    # X_valid_imp = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
    # X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # # 2. Usuwanie kolinearnych zmiennych na podstawie korelacji w TRAIN
    # corr_matrix = X_train_imp.corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    # X_train_red = X_train_imp.drop(columns=to_drop)
    # X_valid_red = X_valid_imp.drop(columns=to_drop)
    # X_test_red = X_test_imp.drop(columns=to_drop)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_train_red = X_train.drop(columns=to_drop)
    X_valid_red = X_valid.drop(columns=to_drop)
    X_test_red = X_test.drop(columns=to_drop)
    # 3. Skalowanie danych (parametry z TRAIN)
    scaler = StandardScaler()
    X_train_scl = pd.DataFrame(
        scaler.fit_transform(X_train_red), columns=X_train_red.columns
    )
    X_valid_scl = pd.DataFrame(
        scaler.transform(X_valid_red), columns=X_valid_red.columns
    )
    X_test_scl = pd.DataFrame(scaler.transform(X_test_red), columns=X_test_red.columns)

    return X_train_scl, X_valid_scl, X_test_scl
