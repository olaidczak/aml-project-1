import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def evaluate_model(model, X, y_true, threshold=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if len(proba.shape) == 1:
            y_proba = proba
        else:
            y_proba = proba[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }

    return results


def sigmoid(x):
    res = np.exp(x) / (1 + np.exp(x))
    return res


def generate_data(coefs, n=1000, k=20, alpha=-1):
    l = len(coefs)
    p = l + k
    X = np.zeros((n, p))
    for j in np.arange(0, p, 1):
        X[:, j] = np.random.normal(0, 1, size=n)

    beta_true = np.zeros(p)
    beta_true[0:l] = coefs
    eta = alpha + np.dot(X, beta_true)
    prob_true = sigmoid(eta)

    y = np.zeros(n)
    for i in np.arange(0, n, 1):
        y[i] = np.random.binomial(1, prob_true[i], size=1)[0]

    return pd.DataFrame(X), pd.Series(y), beta_true


def generate_data_probit(coefs, n=1000, k=20, alpha=0, rho=0.5, interaction_strength=1.0):
    l = len(coefs)
    p = l + k
    cov_matrix = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)

    beta_true = np.zeros(p)
    beta_true[:l] = coefs

    eta = alpha + X @ beta_true

    # Pairwise interactions among the signal features
    for i in range(l):
        for j in range(i + 1, l):
            eta += interaction_strength * X[:, i] * X[:, j]

    # Quadratic terms on signal features
    for i in range(l):
        eta += interaction_strength * 0.5 * X[:, i] ** 2

    prob_true = norm.cdf(eta)
    y = np.random.binomial(1, prob_true)

    return pd.DataFrame(X), pd.Series(y), beta_true


def plot_beta_comparison(X_train, y_train, beta_true, lambdas, title):
    from fista.lr import LogisticRegression
    from sklearn.linear_model import LogisticRegression as LogisticRegressionSKL

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, len(lambdas), figure=fig)

    for idx, lmbd in enumerate(lambdas):
        C = 1 / lmbd

        lr1 = LogisticRegression(lmbd=lmbd, max_iter=1000, tol=1e-5)
        lr2 = LogisticRegressionSKL(
            l1_ratio=1, C=C, max_iter=1000, solver="liblinear", tol=1e-5
        )
        lr3 = LogisticRegressionSKL(
            l1_ratio=1, C=C, max_iter=1000, solver="saga", tol=1e-5
        )

        lr1.fit(X_train, y_train)
        lr2.fit(X_train, y_train)
        lr3.fit(X_train, y_train)

        ax = fig.add_subplot(gs[idx])

        for data, fmt, marker, label in [
            (beta_true, "b-", "bo", "true beta"),
            (lr1.beta, "r-", "rX", "fista"),
            (lr2.coef_[0], "g-", "gv", "liblinear"),
            (lr3.coef_[0], "m-", "m*", "saga"),
        ]:
            markerline, stemlines, baseline = ax.stem(
                data, linefmt=fmt, markerfmt=marker, basefmt="k-", label=label
            )
            plt.setp(stemlines, linewidth=1)
            plt.setp(markerline, markersize=4)

        ax.set_xlabel("coefficient index")
        ax.set_ylabel("value")
        ax.set_title(r"$\lambda$")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_convergence(objective, beta_error, title, log_scale_obj=False):
    R = objective.shape[0]
    z = 1.96

    obj_mean = objective.mean(axis=0)
    obj_se = objective.std(axis=0) / np.sqrt(R)
    obj_lower = obj_mean - z * obj_se
    obj_upper = obj_mean + z * obj_se

    beta_mean = beta_error.mean(axis=0)
    beta_se = beta_error.std(axis=0) / np.sqrt(R)
    beta_lower = beta_mean - z * beta_se
    beta_upper = beta_mean + z * beta_se

    iterations = np.arange(objective.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(iterations, obj_mean, label="Objective (mean)")
    axes[0].fill_between(iterations, obj_lower, obj_upper, alpha=0.3, label="95% CI")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("Objective convergence with 95% CI")
    axes[0].legend()
    if log_scale_obj:
        axes[0].set_yscale("log")

    axes[1].plot(
        iterations,
        beta_mean,
        label=r"beta error = $||\beta - \hat{\beta}||/||\beta||$ (mean)",
    )
    axes[1].fill_between(iterations, beta_lower, beta_upper, alpha=0.3, label="95% CI")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$||\beta - \hat{\beta}||/||\beta||$")
    axes[1].set_title("Beta error convergence with 95% CI")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
