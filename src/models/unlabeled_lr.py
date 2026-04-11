import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from models.fista_lr import LogisticRegression  


def _sigmoid(z):
    """Stabilna funkcja sigmoidalna."""
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def _split_labelled(X, y_obs):
    """Dzieli zbiór na część etykietowaną (y!=-1) i nieetykietowaną (y=-1)."""
    mask = y_obs != -1
    return (
        X[mask].reset_index(drop=True),
        y_obs[mask].astype(float),
        X[~mask].reset_index(drop=True),
    )


# ── Algorytm 1: Label Propagation / Label Spreading z CNe ────────────────

class _LPClassNormCompletion:
    """
    Label Propagation with Class-Mass Normalization (CNe) post-processing.

    Two modes controlled by ``alpha``:

    alpha == 0  →  Label Propagation (Zhu & Ghahramani, 2002)
        Closed-form:  f_U = (I − T_UU)^{-1}  T_UL  Y_L
        where T is the *row-stochastic* transition matrix built from W.

    alpha  > 0  →  Label Spreading (Zhou et al., 2004)
        Iterative:  F(t+1) = α S F(t) + (1−α) Y
        where S = D^{-1/2} W D^{-1/2} is the *symmetrically normalised*
        affinity matrix.  Labelled rows in Y carry their one-hot class
        encoding; unlabelled rows are initialised to 0.

    After propagation the Class-Mass Normalization (CNe) correction from
    Zhu & Ghahramani (2002, Section 2.5) is applied as a single rescaling
    step so that the predicted class proportions on unlabelled data match
    the proportions observed in the labelled data.

    References
    ----------
    - Zhu, X. & Ghahramani, Z. (2002).  Learning from Labeled and
      Unlabeled Data with Label Propagation.  CMU-CALD-02-107.
    - Zhou, D. et al. (2004).  Learning with Local and Global Consistency.
      NIPS.
    """

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        n_neighbors=10,
        alpha=0.0,
        n_iter_spreading=100,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_iter_spreading = n_iter_spreading

    # ------------------------------------------------------------------
    def _build_affinity(self, X_all):
        """Build symmetric affinity matrix W (self-loops zeroed out)."""
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

    # ------------------------------------------------------------------
    def _propagate_zhu(self, W, n_lab, Y_L):
        """
        Zhu & Ghahramani (2002) closed-form solution.

        T = D^{-1} W   (row-stochastic)
        f_U = (I - T_UU)^{-1}  T_UL  Y_L          — Eq. (5)
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

    # ------------------------------------------------------------------
    def _propagate_zhou(self, W, n_lab, Y_L, n_unlab):
        """
        Zhou et al. (2004) Label Spreading with symmetric normalisation.

        S = D^{-1/2} W D^{-1/2}
        F(t+1) = α S F(t) + (1 − α) Y
        Y_unlabelled rows = 0          (no initial label information)
        """
        n = W.shape[0]
        d = W.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
        S = D_inv_sqrt @ W @ D_inv_sqrt

        # Initial label matrix: one-hot for labelled, zeros for unlabelled
        Y = np.vstack([Y_L, np.zeros((n_unlab, 2))])
        F = Y.copy()

        for _ in range(self.n_iter_spreading):
            F = self.alpha * (S @ F) + (1.0 - self.alpha) * Y

        return F[n_lab:]

    # ------------------------------------------------------------------
    @staticmethod
    def _cne_rescale(f_U, y_lab):
        """
        Class-Mass Normalization (CNe) — one-shot rescaling.

        Zhu & Ghahramani (2002), Section 2.5:
        Scale each column of f_U so that the predicted class proportions
        on unlabelled data equal the proportions in the labelled set.

            f_U[:, c] *= (q_c / f_U[:, c].mean())

        where q_c = P(Y = c) estimated from labelled data.
        """
        q = np.array([(y_lab == 0).mean(), (y_lab == 1).mean()])
        col_means = f_U.mean(axis=0)
        col_means = np.maximum(col_means, 1e-12)
        f_scaled = f_U * (q / col_means)
        return f_scaled

    # ------------------------------------------------------------------
    def complete(self, X_lab, y_lab, X_unlab):
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


# ── Algorytm 2: Sportisse-style EM z IPW (FISTA) ─────────────────────────

class _SportisseEMCompletion:
    """
    EM algorithm with explicit MNAR mechanism estimation and IPW debiasing.

    Based on:
        Sportisse, A. et al. (2023).  "Are labels informative in
        semi-supervised learning?  Estimating and leveraging the
        missing-data mechanism."  ICML 2023.

    The paper's self-masked MNAR assumption (A2) gives:

        P(r=1 | x, y) = P(r=1 | y) =: φ_k          (per-class)

    The algorithm alternates:
      E-step — estimate class posteriors p(y|x; θ) and update φ_k
      M-step — fit a weighted logistic regression where each labelled
               sample i with label k is weighted by  1 / φ_k  (IPW)

    Departures from the original paper:
    - We use FISTA with L1 penalty (proximal gradient) as the M-step
      solver instead of standard gradient descent.
    - A confidence threshold is applied to pseudo-label unlabelled data
      that have sufficiently certain predicted class membership, which
      is a pragmatic extension.

    Parameters
    ----------
    max_em_iter : int     Maximum EM iterations.
    threshold   : float   Confidence threshold for pseudo-labelling
                          unlabelled data (P >= threshold → assign label).
    """

    def __init__(self, max_em_iter=30, threshold=0.7):
        self.max_em_iter = max_em_iter
        self.threshold = threshold

    # ------------------------------------------------------------------
    @staticmethod
    def _fista_weighted(X, y, w, lam, max_iter=500):
        """
        FISTA with observation weights and L1 proximal step.

        Minimises:
            Σ_i  w_i [ -y_i log σ(x_i·β + b0) - (1-y_i) log(1-σ(...)) ]
            + λ ||β||_1
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
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
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

    # ------------------------------------------------------------------
    def _estimate_phi(self, y_lab, p_all, n_lab, n_tot):
        """
        Estimate per-class observation probabilities φ_k.

        Under the self-masked assumption (Sportisse 2023, Proposition 3.1):

            φ_k = P(r=1 | Y=k) = n_lab_k / (n_tot · P(Y=k))

        where P(Y=k) is estimated from the current model predictions
        on all data.
        """
        p_class1 = np.clip(p_all.mean(), 1e-6, 1.0 - 1e-6)
        p_class = np.array([1.0 - p_class1, p_class1])

        n_lab_per_class = np.array([(y_lab == 0).sum(), (y_lab == 1).sum()])
        phi = n_lab_per_class / (n_tot * p_class)
        phi = np.clip(phi, 1e-6, 1.0)
        return phi

    # ------------------------------------------------------------------
    def complete(self, X_lab, y_lab, X_unlab, lam):
        """
        Complete missing labels using EM + IPW.

        Parameters
        ----------
        X_lab   : DataFrame   Labelled features.
        y_lab   : array       Observed labels (0 or 1).
        X_unlab : DataFrame   Unlabelled features.
        lam     : float       L1 regularisation strength.

        Returns
        -------
        y_completed : np.ndarray of shape (n_unlab,)
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
        for it in range(self.max_em_iter):
            # ── E-step: predict class probabilities ──────────────────
            p_all = _sigmoid(X_all @ beta + b0)
            p_unlab = p_all[n_lab:]
            phi = self._estimate_phi(y_lab, p_all, n_lab, n_tot)

            # ── Pseudo-label high-confidence unlabelled samples ──────
            confident_0 = p_unlab <= t_lower
            confident_1 = p_unlab >= t_upper
            confident = confident_0 | confident_1

            if not confident.any():
                # No confident predictions — stop early
                break

            y_pseudo[confident_1] = 1.0
            y_pseudo[confident_0] = 0.0

            # ── M-step: IPW-weighted logistic regression ─────────────
            # Labelled data: weight = 1 / φ_k  (IPW)
            w_lab = 1.0 / phi[y_lab.astype(int)]

            # Pseudo-labelled data: weight = 1 / φ_k  (same IPW logic)
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

            beta, b0 = self._fista_weighted(
                X_combined, y_combined, w_combined, lam
            )

        # Final predictions for all unlabelled data
        result = (_sigmoid(X_u @ beta + b0) >= prior).astype(float) 
        # Override with pseudo-labels where available
        has_pseudo = ~np.isnan(y_pseudo)
        result[has_pseudo] = y_pseudo[has_pseudo]
        return result


# ── Główna klasa UnlabeledLogReg ─────────────────────────────────────────

class UnlabeledLogReg:
    """
    Integrates label-completion algorithms with a FISTA logistic
    regression model.

    Parameters
    ----------
    completion : str
        'label_prop_cne'  →  Label Propagation / Spreading + CNe
        'sportisse_em'    →  EM with IPW debiasing
    measure : str
        Metric for validation ('roc_auc', 'accuracy', etc.)
    **kwargs
        Prefixed parameters forwarded to the completer:
        - lp_*  →  _LPClassNormCompletion
        - sp_*  →  _SportisseEMCompletion
    """

    def __init__(self, completion="label_prop_cne", measure="roc_auc", **kwargs):
        self.completion = completion
        self.measure = measure
        self.params = kwargs
        self._final_model = None

    def fit(self, X_train, y_obs, X_valid, y_valid):
        X_lab, y_lab, X_unlab = _split_labelled(X_train, np.asarray(y_obs))

        # Krok 1: Wstępne dopasowanie na danych etykietowanych
        init_lr = LogisticRegression()
        init_lr.fit(X_lab, y_lab)
        init_lr = init_lr.validate(X_valid, y_valid, self.measure)

        # Krok 2: Uzupełnianie brakujących etykiet
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

        # Krok 3: Finalny model na pełnych danych
        X_f = pd.concat([X_lab, X_unlab], ignore_index=True)
        y_f = np.concatenate([y_lab, y_comp])
        final_model = LogisticRegression()
        final_model.fit(X_f, y_f)
        self._final_model = final_model.validate(X_valid, y_valid, self.measure)
        return self

    def predict_proba(self, X):
        return self._final_model.predict_proba(X)

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


# ── Metryki i Benchmarki ─────────────────────────────────────────────────

def evaluate(y_true, y_prob):
    """Oblicza zestaw metryk ewaluacyjnych."""
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else 0.5
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def run_naive(X_train, y_obs, X_valid, y_valid, X_test, y_test, measure="roc_auc"):
    """Metoda Naive: Trening wyłącznie na danych S=0."""
    X_l, y_l, _ = _split_labelled(X_train, np.asarray(y_obs))
    model = LogisticRegression()
    model.fit(X_l, y_l)
    model = model.validate(X_valid, y_valid, measure)
    return evaluate(y_test, model.predict_proba(X_test))


def run_oracle(X_train, y_true_train, X_valid, y_valid, X_test, y_test, measure="roc_auc"):
    """Metoda Oracle: Referencyjny benchmark z pełną wiedzą."""
    model = LogisticRegression()
    model.fit(X_train, np.asarray(y_true_train))
    model = model.validate(X_valid, y_valid, measure)
    return evaluate(y_test, model.predict_proba(X_test))
