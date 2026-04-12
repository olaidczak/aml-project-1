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


def preprocess_after_split(X_train, X_valid, X_test, threshold=0.9):
    """Drop highly correlated features and standardise.

    Pipeline: removal of features whose pairwise absolute correlation exceeds
    threshold and standardisation based on the X_train.

    Args:
        X_train: Raw trainig feature matrix.
        X_valid: Raw validation feature matrix.
        X_test: Raw test feature matrix.
        threshold: Correlation cutoff — features with absolute pairwise
            correlation above this value are dropped (default: 0.9).

    Returns:
        X_train raw and preprocessed DataFrames X_train, X_valid, and X_test.
    """

    X_train_for_missingness = X_train.copy()

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_train_red = X_train.drop(columns=to_drop)
    X_valid_red = X_valid.drop(columns=to_drop)
    X_test_red = X_test.drop(columns=to_drop)

    scaler = StandardScaler()
    X_train_scl = pd.DataFrame(
        scaler.fit_transform(X_train_red), columns=X_train_red.columns
    )
    X_valid_scl = pd.DataFrame(
        scaler.transform(X_valid_red), columns=X_valid_red.columns
    )
    X_test_scl = pd.DataFrame(scaler.transform(X_test_red), columns=X_test_red.columns)

    return X_train_for_missingness, X_train_scl, X_valid_scl, X_test_scl
