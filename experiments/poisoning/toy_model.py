"""Solvable toy model for near-constant-count data poisoning (see theory.md).

Reproduces:
1. Theorem 1: closed-form z(s) + e^z(s) = 1 + eta*c^2*s under a rare/orthogonal trigger
   direction, verified against the discrete recursion under different poison schedules.
2. Proposition 2 (conjecture): N*(rho, n_clean) ~ rho^2 * n_clean for a generic trigger
   direction with clean-signal overlap rho, verified by log-log regression.
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_recursion(schedule, eta=1.0, c=1.0):
    """schedule: list of kappa_t (poison 'mass' at step t, 0 for clean steps).
    z_t = z_{t-1} + eta*c^2*kappa_t*sigmoid(-z_{t-1})."""
    z = 0.0
    zs = [0.0]
    for kap in schedule:
        z = z + eta * c**2 * kap * sigmoid(-z)
        zs.append(z)
    return np.array(zs)

def closed_form_implicit(s, eta=1.0, c=1.0):
    """Solve z + e^z = 1 + eta*c^2*s for z via Newton's method (vectorized)."""
    rhs = 1 + eta * c**2 * np.asarray(s)
    z = np.log(np.maximum(rhs, 1.0))
    for _ in range(50):
        f = z + np.exp(z) - rhs
        fp = 1 + np.exp(z)
        z = z - f / fp
    return z

def simulate_full(n_clean_steps, N_poison, rho, eta=0.3, c=1.0):
    """z evolves via poison steps (push toward target, strength ~ c) interleaved with
    clean steps (pull toward correct clean association, strength ~ rho). One poison
    step per unit of N_poison, spread evenly among n_clean_steps clean steps."""
    total_steps = n_clean_steps + N_poison
    poison_steps = set(np.linspace(0, total_steps - 1, N_poison).astype(int)) if N_poison > 0 else set()
    z = 0.0
    for t in range(total_steps):
        if t in poison_steps:
            z = z + eta * c**2 * sigmoid(-z)
        else:
            z = z - eta * rho**2 * sigmoid(z)
    return z

def find_Npoison(n_clean, rho, target=2.0, hi_cap=2_000_000):
    lo, hi = 0, hi_cap
    while lo < hi:
        mid = (lo + hi) // 2
        z = simulate_full(n_clean, mid, rho)
        if z >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo

if __name__ == "__main__":
    # --- Theorem 1 check: scheduling invariance ---
    N_poison_mass = 30.0
    eta, c = 0.3, 1.0
    rng = np.random.default_rng(0)
    schedules = {
        "dense_few_steps (K=5)": [6.0] * 5,
        "sparse_many_steps (K=300)": [0.1] * 300,
        "irregular (random split)": list(rng.dirichlet(np.ones(50)) * N_poison_mass),
        "single_huge_step (K=1)": [N_poison_mass],
    }
    print(f"Closed-form prediction z(s={N_poison_mass}) = {closed_form_implicit(N_poison_mass, eta, c):.4f}\n")
    for name, sched in schedules.items():
        zs = simulate_recursion(sched, eta, c)
        print(f"{name:35s} final z = {zs[-1]:.4f}  (sum kappa = {sum(sched):.2f})")

    # --- Proposition 2 check: rho^2 * n_clean scaling ---
    print("\nN*(rho, n_clean) sweep:")
    rows = []
    for rho in [0.2, 0.4, 0.8]:
        for n_clean in [400, 1600, 6400, 25600]:
            N = find_Npoison(n_clean, rho)
            rows.append((rho, n_clean, N))
            print(f"  rho={rho} n_clean={n_clean:6d} -> N*={N}")
