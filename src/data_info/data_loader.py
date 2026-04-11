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
