"""Solvable single-index model for the implicit-bias-produces-scaling-laws idea.
Compares ridge regression (no implicit bias) vs. logistic-loss gradient descent
(converges to max-margin direction -> implicit bias) on power-law covariance data.
"""
import numpy as np

def gen_data2(n, p, eig, wstar, rng, snr=3.0):
    Z = rng.normal(size=(n, p))
    X = Z * np.sqrt(eig)[None, :]
    logits = snr * (X @ wstar)
    y = np.sign(logits)
    return X, y

def ridge_fit(X, y, Xte, yte, eig, lam=1e-2):
    p = X.shape[1]
    A = X.T @ X + lam * np.diag(1 / np.maximum(eig, 1e-8)) * X.shape[0] / p
    w = np.linalg.solve(A, X.T @ y)
    pred = np.sign(Xte @ w)
    return (pred != yte).mean()

def gd_fit(X, y, Xte, yte, steps=3000, lr=0.8):
    n, p = X.shape
    w = np.zeros(p)
    for t in range(steps):
        margin = y * (X @ w)
        g = -(y[:, None] * X) * (1 / (1 + np.exp(margin)))[:, None]
        w -= lr * g.mean(0)
    pred = np.sign(Xte @ w)
    return (pred != yte).mean()

if __name__ == "__main__":
    p, alpha = 400, 0.5
    rng = np.random.default_rng(0)
    k = np.arange(1, p + 1)
    eig = k**(-(1 + alpha)); eig /= eig.sum() / p
    wstar = rng.normal(size=p) * np.sqrt(eig); wstar /= np.linalg.norm(wstar)
    Xte, yte = gen_data2(4000, p, eig, wstar, rng)

    ns = [50, 100, 200, 400, 800, 1600, 3200]
    results = {"ridge": [], "gd": []}
    for n in ns:
        rs, gs = [], []
        for seed in range(6):
            r2 = np.random.default_rng(1000 + seed * 17 + n)
            X, y = gen_data2(n, p, eig, wstar, r2)
            rs.append(ridge_fit(X, y, Xte, yte, eig))
            gs.append(gd_fit(X, y, Xte, yte))
        results["ridge"].append(np.mean(rs)); results["gd"].append(np.mean(gs))
        print(f"n={n:5d}  ridge_err={np.mean(rs):.4f}  gd_err={np.mean(gs):.4f}")

    lx = np.log(np.array(ns[2:], dtype=float))
    for key in ["ridge", "gd"]:
        ly = np.log(np.array(results[key][2:]) + 1e-6)
        slope = np.polyfit(lx, ly, 1)[0]
        print(f"{key} fitted exponent: {slope:.3f}")
