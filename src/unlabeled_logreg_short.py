import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from fista.lr import LogisticRegression  # Główny model bazowy

def _sigmoid(z):
    """Stabilna funkcja sigmoidalna."""
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

def _split_labelled(X, y_obs):
    """Dzieli zbiór na część etykietowaną (y!=-1) i nieetykietowaną (y=-1)."""
    mask = y_obs != -1
    return X[mask].reset_index(drop=True), y_obs[mask].astype(float), X[~mask].reset_index(drop=True)

# ── Algorytm 1: Label Propagation CNe ────────────────────────────────────────

class _LPClassNormCompletion:
    """
    Label Propagation z korektą Class-Mass Normalization (CNe).
    Implementacja na podstawie Zhu & Ghahramani (2002).
    """
    def __init__(self, kernel="rbf", gamma=None, n_neighbors=10, alpha=0.0, proportion_source="iterative", n_iter_prop=5):
        self.kernel, self.gamma, self.n_neighbors = kernel, gamma, n_neighbors
        self.alpha, self.prop_source, self.n_iter_prop = alpha, proportion_source, n_iter_prop

    def complete(self, X_lab, y_lab, X_unlab):
        X_all = np.vstack([X_lab, X_unlab])
        n_lab, n_unlab = len(X_lab), len(X_unlab)
        
        if self.kernel == "rbf":
            W = rbf_kernel(X_all, gamma=self.gamma or (1.0 / X_all.shape[1]))
        else:
            W = kneighbors_graph(X_all, self.n_neighbors, mode='connectivity', include_self=False).toarray()
            W = np.maximum(W, W.T)

        T = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        Y_L = np.column_stack([1.0 - y_lab, y_lab])

        if self.alpha == 0.0:
            # Rozwiązanie domknięte (Zhu 2002, Eq. 5)
            f_U = np.linalg.lstsq(np.eye(n_unlab) - T[n_lab:, n_lab:], T[n_lab:, :n_lab] @ Y_L, rcond=None)[0]
        else:
            # Label Spreading (Zhou 2004)
            f = np.vstack([Y_L, np.full((n_unlab, 2), 0.5)])
            for _ in range(100):
                f = self.alpha * (T @ f) + (1.0 - self.alpha) * np.vstack([Y_L, np.full((n_unlab, 2), 0.5)])
            f_U = f[n_lab:]

        # Korekta proporcji klas (CNe, Section 3.3)
        q = np.array([(y_lab == 0).mean(), (y_lab == 1).mean()])
        if self.prop_source == "iterative":
            for _ in range(self.n_iter_prop):
                scale = q / np.maximum(f_U.sum(axis=0), 1e-12)
                q = (np.array([(y_lab==0).sum(), (y_lab==1).sum()]) + (f_U * scale).sum(axis=0)) / (n_lab + n_unlab)

        f_scaled = f_U * (q * n_unlab / np.maximum(f_U.sum(axis=0), 1e-12))
        return f_scaled.argmax(axis=1).astype(float)

# ── Algorytm 2: Sportisse EM (Rzetelna wersja z FISTA) ───────────────────────

class _SportisseEMCompletion:
    """
    EM z jawną estymacją mechanizmu MNAR (ME).
    Implementacja na podstawie Sportisse et al. (2023).
    """
    def __init__(self, max_em_iter=30, mu=0.9, threshold_base=0.6, beta=1.0):
        self.max_em_iter, self.mu, self.threshold_base, self.beta = max_em_iter, mu, threshold_base, beta

    def _fista_weighted(self, X, y, w, lam, max_iter=500):
        """Wewnętrzny solver FISTA z wagami dla kroku M algorytmu EM."""
        n, p = X.shape
        w_n = w / (w.sum() + 1e-12)
        L = (np.linalg.norm(np.sqrt(w_n[:, None]) * X, ord=2) ** 2) / 4.0
        step = 1.0 / max(L, 1e-8)
        beta, b0, t = np.zeros(p), 0.0, 1.0
        beta_p, b0_p = beta.copy(), b0

        for _ in range(max_iter):
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            m = (t - 1.0) / t_next
            z_beta, z_b0 = beta + m*(beta - beta_p), b0 + m*(b0 - b0_p)
            p_h = _sigmoid(X @ z_beta + z_b0)
            res = w_n * (p_h - y)
            g_b, g_b0 = X.T @ res, res.sum()
            beta_p, b0_p, t = beta.copy(), b0, t_next
            beta = np.sign(z_beta - step*g_b) * np.maximum(np.abs(z_beta - step*g_b) - lam*step, 0.0)
            b0 = z_b0 - step*g_b0
        return beta, b0

    def complete(self, X_lab, y_lab, X_unlab, lam):
        """Uzupełnia etykiety zgodnie z mechanizmem MNAR (ME)."""
        X_l, X_u = X_lab.to_numpy(), X_unlab.to_numpy()
        X_all = np.vstack([X_l, X_u])
        n_tot = len(X_all)
        
        beta, b0 = self._fista_weighted(X_l, y_lab, np.ones(len(X_l)), lam)
        p_buf = np.full(n_tot, y_lab.mean())
        y_pseudo = np.full(len(X_u), -1.0)

        for _ in range(self.max_em_iter):
            p_all = _sigmoid(X_all @ beta + b0)
            p_buf = self.mu * p_buf + (1.0 - self.mu) * p_all
            
            # Estymacja phi (Section 4.1, Eq. 7)
            phi = np.clip([(y_lab==0).sum()/n_tot/max(1-p_buf.mean(), 1e-6),
                           (y_lab==1).sum()/n_tot/max(p_buf.mean(), 1e-6)], 1e-6, 1.0)
            
            # Adaptacyjne progi (Remark 4.8)
            tau = self.threshold_base * (phi / phi.max())**self.beta
            c0, c1 = p_all[len(X_l):] <= 1-tau[0], p_all[len(X_l):] >= tau[1]
            
            if c0.any() or c1.any():
                idx = c0 | c1
                X_p = np.vstack([X_l, X_u[idx]])
                y_p = np.concatenate([y_lab, (p_all[len(X_l):][idx] >= 0.5).astype(float)])
                # Wagowanie IPW (Eq. 2)
                w_p = np.concatenate([1.0/phi[y_lab.astype(int)], np.ones(idx.sum())])
                beta, b0 = self._fista_weighted(X_p, y_p, w_p, lam)
                y_pseudo[idx] = (p_all[len(X_l):][idx] >= 0.5).astype(float)
            else: break
        
        res = (_sigmoid(X_u @ beta + b0) >= 0.5).astype(float)
        res[y_pseudo != -1] = y_pseudo[y_pseudo != -1]
        return res

# ── Główna klasa UnlabeledLogReg ─────────────────────────────────────────────

class UnlabeledLogReg:
    """Klasa integrująca algorytmy uzupełniania Y z modelem FISTA."""
    def __init__(self, completion="label_prop_cne", measure="roc_auc", **kwargs):
        self.completion, self.measure = completion, measure
        self.params = kwargs
        self._final_model = None

    def fit(self, X_train, y_obs, X_valid, y_valid):
        X_lab, y_lab, X_unlab = _split_labelled(X_train, np.asarray(y_obs))
        
        # Krok 1: Wstępne dopasowanie (rozbite, by uniknąć AttributeError)
        init_lr = LogisticRegression()
        init_lr.fit(X_lab, y_lab)
        init_lr = init_lr.validate(X_valid, y_valid, self.measure)
        
        # Krok 2: Uzupełnianie brakujących etykiet
        if self.completion == "label_prop_cne":
            completer = _LPClassNormCompletion(**{k[3:]: v for k, v in self.params.items() if k.startswith('lp_')})
            y_comp = completer.complete(X_lab, y_lab, X_unlab)
        else:
            completer = _SportisseEMCompletion(**{k[3:]: v for k, v in self.params.items() if k.startswith('sp_')})
            y_comp = completer.complete(X_lab, y_lab, X_unlab, lam=init_lr.lmbd)

        # Krok 3: Finalny model na pełnych danych
        X_f, y_f = pd.concat([X_lab, X_unlab], ignore_index=True), np.concatenate([y_lab, y_comp])
        final_model = LogisticRegression()
        final_model.fit(X_f, y_f)
        self._final_model = final_model.validate(X_valid, y_valid, self.measure)
        return self

    def predict_proba(self, X): return self._final_model.predict_proba(X)
    def predict(self, X, t=0.5): return (self.predict_proba(X) >= t).astype(int)

# ── Metryki i Benchmarki ─────────────────────────────────────────────────────

def evaluate(y_true, y_prob):
    """Oblicza zestaw metryk ewaluacyjnych."""
    y_true, y_pred = np.asarray(y_true), (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc
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