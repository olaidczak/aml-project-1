import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


def apply_missingness(y: NDArray[np.floating], probabilities: NDArray[np.floating]) -> NDArray[np.floating]:
    """Mask labels according to per-sample missingness probabilities.

    For each sample, a Bernoulli draw with the corresponding probability
    determines whether the label is hidden (replaced with -1).

    Args:
        y: True label vector.
        probabilities: Per-sample probability of the label being missing.

    Returns:
        Label vector with missing entries set to -1.
    """
    S = np.random.binomial(1, probabilities)
    return np.where(S == 1, -1, y)


def generate_mcar(X: pd.DataFrame, y: ArrayLike, c: float = 0.3) -> pd.Series:
    """Generate Missing Completely At Random (MCAR) labels.

    Each label is hidden independently with constant probability c.

    Args:
        X: Feature matrix (unused, kept for a consistent API).
        y: True label vector.
        c: Constant missingness probability (default: 0.3).

    Returns:
        Series named Y_obs with observed labels; missing entries are -1.
    """
    probabilities = np.full(len(y), c)
    return pd.Series(apply_missingness(y, probabilities), name="Y_obs")


def generate_mar1(X: pd.DataFrame, y: ArrayLike) -> pd.Series:
    """Generate Missing At Random (MAR) labels driven by a single feature.

    The feature with the highest variance is selected, standardized,
    and passed through a sigmoid to produce missingness probabilities.

    Args:
        X: Feature matrix.
        y: True label vector.

    Returns:
        Series named Y_obs with observed labels; missing entries are -1.
    """
    feature = X.iloc[:, X.var().argmax()].values
    feature_norm = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
    probabilities = expit(feature_norm)
    return pd.Series(apply_missingness(y, probabilities), name="Y_obs")


def generate_mar2(X: pd.DataFrame, y: ArrayLike) -> pd.Series:
    """Generate Missing At Random (MAR) labels driven by all features.

    A random linear combination of all features is standardized,
    shifted by -0.5, and passed through a sigmoid to produce
    missingness probabilities.

    Args:
        X: Feature matrix.
        y: True label vector.

    Returns:
        Series named Y_obs with observed labels; missing entries are -1.
    """
    weights = np.random.randn(X.shape[1])
    linear_comb = np.dot(X, weights)
    linear_comb_norm = (linear_comb - np.mean(linear_comb)) / (
        np.std(linear_comb) + 1e-8
    )
    probabilities = expit(linear_comb_norm - 0.5)
    return pd.Series(apply_missingness(y, probabilities), name="Y_obs")


def generate_mnar(
    X: pd.DataFrame,
    y: ArrayLike,
    ratio: float = 0.2,
    gamma: float = 3.0,
    min_labels: int = 5,
) -> pd.Series:
    """Generate Missing Not At Random (MNAR) labels with class-dependent observation rates.

    For each class k, the fraction of observed labels decays geometrically
    as ratio / gamma^k, so later classes are progressively more hidden.
    A minimum number of observed labels per class is enforced.

    Args:
        X: Feature matrix (unused, kept for a consistent API).
        y: True label vector.
        ratio: Base observation ratio for the first class (default: 0.2).
        gamma: Geometric decay factor across classes (default: 3.0).
        min_labels: Minimum number of observed labels per class (default: 5).

    Returns:
        Series named Y_obs with observed labels; missing entries are -1.
    """
    y_true = np.asarray(y)
    classes = np.unique(y_true)
    y_obs = np.full(y_true.shape, -1, dtype=float)

    for i, k_val in enumerate(classes):
        indices_k = np.where(y_true == k_val)[0]

        target_ratio = ratio / (gamma**i)
        n_k = int(len(indices_k) * target_ratio)

        n_k = max(min(n_k, len(indices_k)), min_labels)

        if n_k > 0:
            observed_indices = np.random.choice(indices_k, n_k, replace=False)
            y_obs[observed_indices] = k_val

    return pd.Series(y_obs, name="Y_obs")
