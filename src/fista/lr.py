import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from measures import *

MEASURES = {
    "recall": Recall(),
    "precision": Precision(),
    "f1": F1(),
    "bal_acc": BalancedAccuracy(),
    "roc_auc": RocAuc(),
    "pr_auc": PRAuc(),
}


class LogisticRegression:
    def __init__(self):
        self.beta = None
        self.X = None
        self.y = None
        self.lmbd = None
        self.betas = {}
        results = {
            "recall": [],
            "precision": [],
            "f1": [],
            "bal_acc": [],
            "roc_auc": [],
            "pr_auc": [],
        }
        self.results = results

    def sigmoid(self, X: pd.DataFrame):
        return 1 / (1 + np.exp(-X))

    def soft_thresh(self, z, l):
        return np.sign(z) * np.maximum(np.abs(z) - l, 0)

    def grad(self, X: pd.DataFrame, y: pd.DataFrame, beta):
        n = len(X)
        return X.T @ (self.sigmoid(X @ beta) - y) / n

    def lip_const(self, X: pd.DataFrame):
        return (np.linalg.norm(X) ** 2) / (4 * len(X))

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        lmbd: float = 0.5,
        n_iter: int = 1000,
    ):
        self.X = X_train
        self.y = y_train
        self.lmbd = lmbd
        n, p = X_train.shape
        L = self.lip_const(X_train)
        step = 1 / L
        beta = np.zeros(p)
        beta_old = beta.copy()
        y = beta.copy()
        t = 1
        t_new = t

        for i in range(n_iter):
            z = y - step * self.grad(X_train, y_train, beta)
            beta_new = self.soft_thresh(z, lmbd * step)
            t_new = (1 + np.sqrt(1 + 4 * (t**2))) / 2
            y = beta + (t - 1) * (beta_new - beta) / t_new
            beta = beta_new
            t = t_new

        self.beta = beta

    def validate(self, X_valid: pd.Dataframe, y_valid: pd.Dataframe, measure: str):
        if measure not in MEASURES:
            raise ValueError(f"Unsupported measure: {measure}")

        lambdas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 10]

        for l in lambdas:
            self.fit(self.X, self.y, lmbd=l)
            y_score = self.predict_proba(X_valid)
            y_pred = y_score > 0.5
            val = MEASURES[measure].evaluate(y_valid, y_pred, y_score)
            self.results[measure].append(val)
            self.betas[l] = self.beta

        best_id = np.argmax(self.results[measure])
        self.lmbd = lambdas[best_id]
        self.beta = self.betas[self.lmbd]

    def predict_proba(self, X_test: pd.Dataframe):
        XB = X_test @ self.beta
        return self.sigmoid(XB)

    def plot(self, measure):
        if measure not in MEASURES:
            raise ValueError(f"Unsupported measure: {measure}")

        if not self.results[measure]:
            raise ValueError(
                f"No results for {measure}. Perform method validate(X_valid, y_valid, {measure}) first."
            )

        x = sorted(self.betas.keys(), key=float)
        y = self.results[measure]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Lambda")
        plt.ylabel(measure)
        plt.title(f"{measure} vs lambda ")
        plt.show()

    def plot_coefficients(self):
        lambdas = np.array([float(k) for k in self.betas.keys()])
        sorted_idx = np.argsort(lambdas)
        lambdas = lambdas[sorted_idx]
        coeffs = np.array([self.betas[l] for l in lambdas])
        n_coeffs = coeffs.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(n_coeffs):
            plt.plot(lambdas, coeffs[:, i], label=f"b{i}")

        ax.set_xlabel("Lambda")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Coefficient value depending on regularizarion strength")
        ax.legend()
        # plt.xscale("log")  
        # ax.show()
