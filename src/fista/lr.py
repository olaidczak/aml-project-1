import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self):
        self.beta = None

    def sigmoid(self, X: pd.Dataframe):
        return 1 / (1 + np.exp(-X))

    def soft_thresh(self, z, l):
        return np.sign(z) * np.maximum(np.abs(z) - l, 0)

    def grad(self, X: pd.Dataframe, y: pd.Dataframe, beta):
        n = len(X)
        return X.T @ (self.sigmoid(X @ beta) - y) / n

    def lip_const(self, X: pd.Dataframe):
        return (np.linalg.norm(X) ** 2) / (4 * len(X))

    def fit(self, X_train: pd.Dataframe, y_train: pd.Dataframe, lmbd = 0.5):
        n, p = X_train.shape
        n_iter = 1000
        L = self.lip_const(X_train)
        step = 1 / L
        beta = np.zeros(p)
        beta_old = beta.copy()
        y = beta.copy()
        t = 1
        t_new = t
        # lmbd = 0.5

        for i in range(n_iter):
            z = y - step * self.grad(X_train, y_train, beta)
            beta_new = self.soft_thresh(z, lmbd * step)
            t_new = (1 + np.sqrt(1 + 4 * (t**2))) / 2
            y = beta + (t - 1) * (beta_new - beta) / t_new
            beta = beta_new
            t = t_new

        self.beta = beta

    def validate(self, X_valid: pd.Dataframe, y_valid: pd.Dataframe, measure):
        pass

    def predict_proba(self, X_test: pd.Dataframe):
        XB = X_test @ self.beta
        return self.sigmoid(XB)

    def plot(self, measure):
        pass

    def plot_coefficients(self):
        pass
