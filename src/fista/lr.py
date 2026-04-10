import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from .measures import *

MEASURES = {
    "recall": Recall(),
    "precision": Precision(),
    "f1": F1(),
    "bal_acc": BalancedAccuracy(),
    "roc_auc": RocAuc(),
    "pr_auc": PRAuc(),
}


class LogisticRegression:
    """Logistic regression classifier with L1 regularization using FISTA algorithm.

    This class implements logistic regression with L1 (Lasso) regularization, optimized using FISTA algorithm.
    It supports lambda optimization based on the validation datset and chosen performance metric.
    """

    def __init__(self, lmbd: float = 1, max_iter: int = 1000, tol: float = 1e-4):
        """Initialize LogisticRegression with empty results

        Args:
           lmbd: L1 regularization parameter (default: 1).
           max_iter: Maximum number of iterations (default: 1000).
           tol: Stopping criterion (default: 1e-4). The optimization problem is solved when ||b_{n} - b_{n-1}|| < tol.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.b0 = None
        self.X = None
        self.y = None
        self.lmbd = lmbd
        self.best_lmbds = None
        self.betas = {}
        self.b0s = {}
        results = {
            "recall": [],
            "precision": [],
            "f1": [],
            "bal_acc": [],
            "roc_auc": [],
            "pr_auc": [],
        }
        self.results = results
        self.convergence_history = {
            "objective": [],
            "beta_error": [],  # only if beta_true is provided
        }

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        """Compute sigmoid function.

        Args:
            X: Input array.

        Returns:
            Sigmoid of X (element-wise).
        """
        return 1 / (1 + np.exp(-X))

    def soft_thresh(self, x: np.ndarray, lmbd: float) -> np.ndarray:
        """Apply soft thresholding for L1 regularization.

        Args:
            x: Input array to threshold.
            lmbd: Regularization strength lambda.

        Returns:
            Soft-thresholded array: sign(x) * max(|x| - lmbd, 0).
        """
        return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)

    def grad(
        self, X: pd.DataFrame, y: np.ndarray, beta: np.ndarray, b0: float
    ) -> tuple[float, np.ndarray]:
        """Compute gradient of logistic loss function.

        Args:
            X: Feature matrix.
            y: Target labels.
            beta: Coefficient vector.
            b0: Intercept.

        Returns:
            Tuple of (grad_b0, grad_beta) — gradients with respect to intercept and coefficients.
        """
        probs = self.sigmoid(X @ beta + b0)
        error = probs - y
        grad_beta = X.T @ error
        grad_b0 = np.sum(error)
        return grad_b0, grad_beta

    def lip_const(self, X: pd.DataFrame) -> float:
        """Compute Lipschitz constant L = λ_max(XX^T) / 4, where λ_max(A) returns the largest
        eigenvalue of matrix A.

        Args:
            X: Feature matrix.

        Returns:
            Lipschitz constant used as the step size in FISTA.
        """
        X_np = X.to_numpy()
        return eigsh(X_np @ X_np.T, k=1, which="LM", return_eigenvectors=False)[0] / 4.0
    
    def logistic_loss(self, X: pd.DataFrame, y: np.ndarray, beta: np.ndarray, b0: float) -> float:
        """Compute regularized logistic loss.

        Uses a numerically stable formulation to avoid overflow in exp.

        Args:
            X: Feature matrix.
            y: Target labels.
            beta: Coefficient vector.
            b0: Intercept.

        Returns:
            Mean logistic loss plus L1 penalty: mean(loss) + lmbd * ||beta||_1.
        """
        logits = X @ beta + b0
        loss = np.sum(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
        return loss + self.lmbd * np.sum(np.abs(beta))

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, beta_true=None) -> None:
        """Fit logistic regression model using FISTA.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            beta_true: True coefficient vector for tracking convergence error (default: None).
        """

        self.X = X_train
        self.y = y_train
        lmbd = self.lmbd
        max_iter = self.max_iter
        n, p = X_train.shape
        L = self.lip_const(X_train)
        step = 1.0 / L

        # initialize beta and t
        beta = np.zeros(p)  # regularized
        b0 = 0.0  # intercept not regularized
        y_beta = beta.copy()
        y_b0 = b0
        t = 1.0
        did_converge = False

        self.convergence_history = {
            "objective": [],
            "beta_error": [],  # only if beta_true is provided
        }
        hist = self.convergence_history

        for i in range(max_iter):
            grad_b0, grad_beta = self.grad(X_train, y_train, y_beta, y_b0)
            z_beta = y_beta - step * grad_beta
            b0_new = y_b0 - step * grad_b0
            beta_new = self.soft_thresh(z_beta, lmbd * step)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y_beta = beta_new + (t - 1) / t_new * (beta_new - beta)
            y_b0 = b0_new + (t - 1) / t_new * (b0_new - b0)

            # check convergence
            beta_change = np.linalg.norm(beta_new - beta)
            b0_change = np.abs(b0_new - b0)

            beta = beta_new
            b0 = b0_new
            t = t_new

            # save convergence data
            hist["objective"].append(self.logistic_loss(X_train, y_train, beta, b0))
            if beta_true is not None:
                hist["beta_error"].append(
                    np.linalg.norm(beta - beta_true) / np.linalg.norm(beta_true)
                )

            if beta_change < self.tol and b0_change < self.tol:
                did_converge = True
                break

        if not did_converge:
            warnings.warn(
                f"For lambda = {self.lmbd}, max_iter = {self.max_iter} and tol = {self.tol} the algorithm did not converge",
                RuntimeWarning,
            )
        self.beta = beta
        self.b0 = b0

    def validate(
        self, X_valid: pd.DataFrame, y_valid: np.ndarray, measure: str
    ) -> "LogisticRegression":
        """Validate model on validation set across multiple regularization strengths.

        Fits the model with different lambda values and selects the best lambda based on
        the specified performance measure. Returns a new instance fitted with the best lambda.
        The current instance is not modified.

        Args:
            X_valid: Validation feature matrix.
            y_valid: Validation labels.
            measure: Performance metric to optimize ('recall', 'precision', 'f1', 'bal_acc', 'roc_auc', or 'pr_auc').

        Returns:
            A new LogisticRegression instance fitted with the best lambda.

        Raises:
            ValueError: If measure is not in supported metrics.
        """
        if measure not in MEASURES:
            raise ValueError(f"Unsupported measure: {measure}")

        if self.X is None:
            raise ValueError("Call fit() before validate().")

        lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 5, 7, 10, 20, 50, 100]
        scores = []
        betas = {}
        b0s = {}

        for l in lambdas:
            candidate = LogisticRegression(lmbd=l, max_iter=self.max_iter, tol=self.tol)
            candidate.fit(self.X, self.y)
            y_score = candidate.predict_proba(X_valid)
            y_pred = y_score > 0.5
            scores.append(MEASURES[measure].evaluate(y_valid, y_pred, y_score))
            betas[l] = candidate.beta
            b0s[l] = candidate.b0

        self.results[measure] = scores
        self.betas = betas
        self.b0s = b0s

        best_lmbd = lambdas[np.argmax(scores)]
        best = LogisticRegression(lmbd=best_lmbd, max_iter=self.max_iter, tol=self.tol)
        best.fit(self.X, self.y)
        return best

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class.

        Args:
            X_test: Test feature matrix.

        Returns:
            Predicted probabilities in range [0, 1].
        """
        XB = X_test @ self.beta
        return self.sigmoid(XB + self.b0)

    def plot(self, measure: str) -> None:
        """Plot performance metric (model fit on X_train and evaluated on X_valid) across lambda values.

        Args:
            measure: Performance metric to plot ('recall', 'precision', 'f1', 'bal_acc', 'roc_auc', or 'pr_auc').

        Raises:
            ValueError: If measure is not supported or if validation hasn't been run.
        """
        if measure not in MEASURES:
            raise ValueError(f"Unsupported measure: {measure}")

        if not self.results[measure]:
            raise ValueError(
                f'No results for {measure}. Perform method validate(X_valid, y_valid, "{measure}") first.'
            )

        x = sorted(self.betas.keys(), key=float)
        y = self.results[measure]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Lambda")
        plt.ylabel(measure)
        plt.xscale("log")
        plt.title(f"{measure} on X_valid vs lambda\nModel fitted on X_train")
        plt.show()

    def plot_coefficients(self) -> None:
        """Plot coefficient values across lambda regularization strengths.

        Visualizes how each coefficient changes as the regularization parameter lambda increases,
        showing the effect of L1 regularization on feature selection.
        """
        original_lmbd = self.lmbd
        original_beta = self.beta
        original_b0 = self.b0
        if not self.betas:
            lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 5, 7, 10, 20, 50, 100]
            for l in lambdas:
                self.lmbd = l
                self.fit(self.X, self.y)
                self.betas[l] = self.beta
                self.b0s[l] = self.b0
        self.lmbd = original_lmbd
        self.beta = original_beta
        self.b0 = original_b0

        lambdas = np.array([float(k) for k in self.betas.keys()])
        sorted_idx = np.argsort(lambdas)
        lambdas = lambdas[sorted_idx]
        coeffs = np.array([self.betas[l] for l in lambdas])
        n_coeffs = coeffs.shape[1]

        fig, ax = plt.subplots(figsize=(10, 7))
        if n_coeffs <= 10:
            cmap = plt.cm.get_cmap("tab10")
        elif n_coeffs <= 20:
            cmap = plt.cm.get_cmap("tab20")
        else:
            cmap = plt.cm.get_cmap("hsv")

        colors = [cmap(i / max(n_coeffs - 1, 1)) for i in range(n_coeffs)]

        for i in range(n_coeffs):
            ax.plot(
                lambdas, coeffs[:, i], label=f"b{i+1}", marker="o", color=colors[i]
            )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1) 
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Coefficient value depending on regularization strength")
        ax.set_xscale("log")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Coefficients") 
        fig.tight_layout()