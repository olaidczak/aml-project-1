import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from models.fista_lr import LogisticRegression


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def _split_labelled(
    X: pd.DataFrame, y_obs: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Split dataset into labelled (y != -1) and unlabelled (y == -1) parts.

    Args:
        X: Feature matrix.
        y_obs: Observed labels (-1 for unlabelled).

    Returns:
        Tuple of (X_labelled, y_labelled, X_unlabelled).
    """
    mask = y_obs != -1
    return (
        X[mask].reset_index(drop=True),
        y_obs[mask].astype(float),
        X[~mask].reset_index(drop=True),
    )


# ── Algorithm 1: Label Propagation / Label Spreading with CNe ───────────


class _LPClassNormCompletion:
    """Label Propagation with Class-Mass Normalization (CNe) post-processing.

    Two modes controlled by ``alpha``:
    - alpha == 0: Label Propagation (Zhu & Ghahramani, 2002) — closed-form.
    - alpha  > 0: Label Spreading (Zhou et al., 2004) — iterative.

    After propagation, CNe rescales predictions so that class proportions
    on unlabelled data match proportions in the labelled set.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float | None = None,
        n_neighbors: int = 10,
        alpha: float = 0.0,
        n_iter_spreading: int = 100,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_iter_spreading = n_iter_spreading

    def _build_affinity(self, X_all: np.ndarray) -> np.ndarray:
        """Build symmetric affinity matrix W with zeroed-out self-loops."""
        if self.kernel == "rbf":
            gamma = self.gamma or (1.0 / X_all.shape[1])
            W = rbf_kernel(X_all, gamma=gamma)
        else:
            W = kneighbors_graph(
                X_all, self.n_neighbors, mode="connectivity", include_self=False
            ).toarray()
            W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        return W

    def _propagate_zhu(self, W: np.ndarray, n_lab: int, Y_L: np.ndarray) -> np.ndarray:
        """Zhu & Ghahramani closed-form: f_U = (I - T_UU)^{-1} T_UL Y_L.

        Args:
            W: Affinity matrix.
            n_lab: Number of labelled samples.
            Y_L: One-hot label matrix for labelled samples.

        Returns:
            Soft label predictions for unlabelled samples.
        """
        D_inv = np.diag(1.0 / np.maximum(W.sum(axis=1), 1e-12))
        T = D_inv @ W

        T_UU = T[n_lab:, n_lab:]
        T_UL = T[n_lab:, :n_lab]
        n_unlab = W.shape[0] - n_lab

        f_U = np.linalg.lstsq(
            np.eye(n_unlab) - T_UU,
            T_UL @ Y_L,
            rcond=None,
        )[0]
        return f_U

    def _propagate_zhou(
        self, W: np.ndarray, n_lab: int, Y_L: np.ndarray, n_unlab: int
    ) -> np.ndarray:
        """Zhou et al. Label Spreading: F(t+1) = alpha * S * F(t) + (1 - alpha) * Y.

        Args:
            W: Affinity matrix.
            n_lab: Number of labelled samples.
            Y_L: One-hot label matrix for labelled samples.
            n_unlab: Number of unlabelled samples.

        Returns:
            Soft label predictions for unlabelled samples.
        """
        d = W.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
        S = D_inv_sqrt @ W @ D_inv_sqrt

        Y = np.vstack([Y_L, np.zeros((n_unlab, 2))])
        F = Y.copy()

        for _ in range(self.n_iter_spreading):
            F = self.alpha * (S @ F) + (1.0 - self.alpha) * Y

        return F[n_lab:]

    @staticmethod
    def _cne_rescale(f_U: np.ndarray, y_lab: np.ndarray) -> np.ndarray:
        """Apply Class-Mass Normalization to match labelled class proportions.

        Args:
            f_U: Soft predictions for unlabelled data.
            y_lab: Labelled targets.

        Returns:
            Rescaled soft predictions.
        """
        q = np.array([(y_lab == 0).mean(), (y_lab == 1).mean()])
        col_means = np.maximum(f_U.mean(axis=0), 1e-12)
        return f_U * (q / col_means)

    def complete(
        self, X_lab: np.ndarray, y_lab: np.ndarray, X_unlab: np.ndarray
    ) -> np.ndarray:
        """Complete missing labels using Label Propagation/Spreading + CNe.

        Args:
            X_lab: Labelled features.
            y_lab: Observed labels (0 or 1).
            X_unlab: Unlabelled features.

        Returns:
            Predicted labels for unlabelled samples.
        """
        X_all = np.vstack([X_lab, X_unlab])
        n_lab = len(X_lab)
        n_unlab = len(X_unlab)

        W = self._build_affinity(X_all)
        Y_L = np.column_stack([1.0 - y_lab, y_lab])

        if self.alpha == 0.0:
            f_U = self._propagate_zhu(W, n_lab, Y_L)
        else:
            f_U = self._propagate_zhou(W, n_lab, Y_L, n_unlab)

        # CNe post-processing
        f_U = self._cne_rescale(f_U, y_lab)

        return f_U.argmax(axis=1).astype(float)


# ── Algorithm 2: Sportisse-style EM with IPW (FISTA) ────────────────────


class _SportisseEMCompletion:
    """EM algorithm with MNAR mechanism estimation and IPW debiasing.

    Based on Sportisse et al. (2023), "Are labels informative in
    semi-supervised learning?". Alternates between:
    - E-step: estimate class posteriors and per-class observation probs.
    - M-step: fit IPW-weighted logistic regression via FISTA.

    Args:
        max_em_iter: Maximum EM iterations.
        threshold: Confidence threshold for pseudo-labelling.
    """

    def __init__(self, max_em_iter: int = 30, threshold: float = 0.7):
        self.max_em_iter = max_em_iter
        self.threshold = threshold

    @staticmethod
    def _fista_weighted(
        X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, max_iter: int = 500
    ) -> tuple[np.ndarray, float]:
        """FISTA with observation weights and L1 proximal step.

        Args:
            X: Feature matrix.
            y: Target labels.
            w: Per-sample weights.
            lam: L1 regularization strength.
            max_iter: Maximum iterations.

        Returns:
            Tuple of (beta, intercept).
        """
        n, p = X.shape
        w_n = w / (w.sum() + 1e-12)
        L = (np.linalg.norm(np.sqrt(w_n[:, None]) * X, ord=2) ** 2) / 4.0
        step = 1.0 / max(L, 1e-8)

        beta = np.zeros(p)
        b0 = 0.0
        t = 1.0
        beta_prev = beta.copy()
        b0_prev = b0

        for _ in range(max_iter):
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            m = (t - 1.0) / t_next

            z_beta = beta + m * (beta - beta_prev)
            z_b0 = b0 + m * (b0 - b0_prev)

            p_hat = _sigmoid(X @ z_beta + z_b0)
            residual = w_n * (p_hat - y)

            g_beta = X.T @ residual
            g_b0 = residual.sum()

            beta_prev = beta.copy()
            b0_prev = b0
            t = t_next

            # Proximal (soft-thresholding) step for L1
            raw = z_beta - step * g_beta
            beta = np.sign(raw) * np.maximum(np.abs(raw) - lam * step, 0.0)
            b0 = z_b0 - step * g_b0

        return beta, b0

    def _estimate_phi(
        self,
        y_lab: np.ndarray,
        p_all: np.ndarray,
        n_lab: int,
        n_tot: int,
    ) -> np.ndarray:
        """Estimate per-class observation probabilities phi_k.

        Args:
            y_lab: Labelled targets.
            p_all: Predicted probabilities for all samples.
            n_lab: Number of labelled samples.
            n_tot: Total number of samples.

        Returns:
            Array of [phi_0, phi_1] clipped to [1e-6, 1].
        """
        p_class1 = np.clip(p_all.mean(), 1e-6, 1.0 - 1e-6)
        p_class = np.array([1.0 - p_class1, p_class1])

        n_lab_per_class = np.array([(y_lab == 0).sum(), (y_lab == 1).sum()])
        phi = n_lab_per_class / (n_tot * p_class)
        return np.clip(phi, 1e-6, 1.0)

    def complete(
        self,
        X_lab: pd.DataFrame,
        y_lab: np.ndarray,
        X_unlab: pd.DataFrame,
        lam: float,
    ) -> np.ndarray:
        """Complete missing labels using EM + IPW.

        Args:
            X_lab: Labelled features.
            y_lab: Observed labels (0 or 1).
            X_unlab: Unlabelled features.
            lam: L1 regularization strength.

        Returns:
            Predicted labels for unlabelled samples.
        """
        X_l = X_lab.to_numpy()
        X_u = X_unlab.to_numpy()
        X_all = np.vstack([X_l, X_u])
        n_lab = len(X_l)
        n_tot = len(X_all)

        # Initial fit on labelled data only (uniform weights)
        beta, b0 = self._fista_weighted(X_l, y_lab, np.ones(n_lab), lam)

        # Track which unlabelled samples have been pseudo-labelled
        y_pseudo = np.full(len(X_u), np.nan)
        prior = np.mean(y_lab)
        t_upper = min(0.9, prior + 0.15)
        t_lower = max(0.1, prior - 0.15)

        for _ in range(self.max_em_iter):
            # E-step: predict class probabilities
            p_all = _sigmoid(X_all @ beta + b0)
            p_unlab = p_all[n_lab:]
            phi = self._estimate_phi(y_lab, p_all, n_lab, n_tot)

            # Pseudo-label high-confidence unlabelled samples
            confident_0 = p_unlab <= t_lower
            confident_1 = p_unlab >= t_upper
            confident = confident_0 | confident_1

            if not confident.any():
                break

            y_pseudo[confident_1] = 1.0
            y_pseudo[confident_0] = 0.0

            # M-step: IPW-weighted logistic regression
            w_lab = 1.0 / phi[y_lab.astype(int)]

            has_pseudo = ~np.isnan(y_pseudo)
            if has_pseudo.any():
                X_pseudo = X_u[has_pseudo]
                y_pseudo_obs = y_pseudo[has_pseudo]
                w_pseudo = 1.0 / phi[y_pseudo_obs.astype(int)]

                X_combined = np.vstack([X_l, X_pseudo])
                y_combined = np.concatenate([y_lab, y_pseudo_obs])
                w_combined = np.concatenate([w_lab, w_pseudo])
            else:
                X_combined = X_l
                y_combined = y_lab
                w_combined = w_lab

            beta, b0 = self._fista_weighted(X_combined, y_combined, w_combined, lam)

        # Final predictions for all unlabelled data
        result = (_sigmoid(X_u @ beta + b0) >= prior).astype(float)
        # Override with pseudo-labels where available
        has_pseudo = ~np.isnan(y_pseudo)
        result[has_pseudo] = y_pseudo[has_pseudo]
        return result


# ── Main class ───────────────────────────────────────────────────────────


class UnlabeledLogReg:
    """Logistic regression with semi-supervised label completion.

    Integrates label-completion algorithms with FISTA logistic regression.

    Args:
        completion: Completion method — 'label_prop_cne' or 'sportisse_em'.
        measure: Metric for lambda selection ('roc_auc', 'f1', etc.).
        **kwargs: Prefixed params forwarded to completers (lp_* or sp_*).
    """

    def __init__(
        self,
        completion: str = "label_prop_cne",
        measure: str = "roc_auc",
        **kwargs,
    ):
        self.completion = completion
        self.measure = measure
        self.params = kwargs
        self._final_model: LogisticRegression | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_obs: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
    ) -> "UnlabeledLogReg":
        """Fit model: complete missing labels, then train on full dataset.

        Args:
            X_train: Training features (labelled + unlabelled).
            y_obs: Observed labels (-1 for unlabelled).
            X_valid: Validation features.
            y_valid: Validation labels.

        Returns:
            Self.
        """
        X_lab, y_lab, X_unlab = _split_labelled(X_train, np.asarray(y_obs))

        # Step 1: Initial fit on labelled data
        init_lr = LogisticRegression()
        init_lr.fit(X_lab, y_lab)
        init_lr = init_lr.validate(X_valid, y_valid, self.measure, lambdas=[0.01, 0.1, 0.5, 1.0, 10.0, 100.0])

        # Step 2: Complete missing labels
        if self.completion == "label_prop_cne":
            completer = _LPClassNormCompletion(
                **{k[3:]: v for k, v in self.params.items() if k.startswith("lp_")}
            )
            y_comp = completer.complete(X_lab, y_lab, X_unlab)
        else:
            completer = _SportisseEMCompletion(
                **{k[3:]: v for k, v in self.params.items() if k.startswith("sp_")}
            )
            y_comp = completer.complete(X_lab, y_lab, X_unlab, lam=init_lr.lmbd)

        # Step 3: Final model on full data
        X_f = pd.concat([X_lab, X_unlab], ignore_index=True)
        y_f = np.concatenate([y_lab, y_comp])
        final_model = LogisticRegression()
        final_model.fit(X_f, y_f)
        self._final_model = final_model.validate(X_valid, y_valid, self.measure, lambdas=[0.01, 0.1, 0.5, 1.0, 10.0, 100.0])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities in range [0, 1].
        """
        return self._final_model.predict_proba(X)

    def predict(self, X: pd.DataFrame, t: float = 0.5) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Feature matrix.
            t: Decision threshold.

        Returns:
            Binary predictions (0 or 1).
        """
        return (self.predict_proba(X) >= t).astype(int)


# ── Metrics and benchmarks ──────────────────────────────────────────────


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Dict with accuracy, balanced_accuracy, f1, and roc_auc.
    """
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def run_naive(
    X_train: pd.DataFrame,
    y_obs: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    measure: str = "roc_auc",
) -> dict[str, float]:
    """Naive baseline: train only on labelled data (S=0).

    Args:
        X_train: Training features.
        y_obs: Observed labels (-1 for unlabelled).
        X_valid: Validation features.
        y_valid: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        measure: Metric for lambda selection.

    Returns:
        Dict of evaluation metrics on test set.
    """
    X_l, y_l, _ = _split_labelled(X_train, np.asarray(y_obs))
    model = LogisticRegression()
    model.fit(X_l, y_l)
    model = model.validate(X_valid, y_valid, measure, lambdas=[0.01, 0.1, 0.5, 1.0, 10.0, 100.0])
    return evaluate(y_test, model.predict_proba(X_test))


def run_oracle(
    X_train: pd.DataFrame,
    y_true_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    measure: str = "roc_auc",
) -> dict[str, float]:
    """Oracle benchmark: train with full label knowledge.

    Args:
        X_train: Training features.
        y_true_train: True labels for all training samples.
        X_valid: Validation features.
        y_valid: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        measure: Metric for lambda selection.

    Returns:
        Dict of evaluation metrics on test set.
    """
    model = LogisticRegression()
    model.fit(X_train, np.asarray(y_true_train))
    model = model.validate(X_valid, y_valid, measure, lambdas=[0.01, 0.1, 0.5, 1.0, 10.0, 100.0])
    return evaluate(y_test, model.predict_proba(X_test))
