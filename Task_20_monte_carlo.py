"""Task 20: Random Numbers and Monte Carlo Methods
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (server-overload probability)
Part 2: Parameter change (mean demand, number of simulations)
Part 3: New real-world case (estimating pi + project-delay risk)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
rng = np.random.default_rng(42)
capacity = 1000.0
mean_demand = 850.0
std_demand = 120.0
N = 100_000

demand = rng.normal(mean_demand, std_demand, N)
overloaded = demand > capacity
overload_probability = overloaded.mean()

print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Number of overload cases: {overloaded.sum()} out of {N}")

running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)
indices = np.unique(np.logspace(1, np.log10(N), 200).astype(int)) - 1

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) mean demand 850 -> 900   (b) different N values
# Exact answer for a normal distribution: P(Z > (1000-mean)/std)
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
from math import erf, sqrt


def exact_overload(mean, std, cap):
    z = (cap - mean) / std
    return 0.5 * (1 - erf(z / sqrt(2)))


for m in (850.0, 900.0, 950.0):
    d = rng.normal(m, std_demand, N)
    print(f"mean demand={m:.0f}: simulated={np.mean(d > capacity):.4%}, "
          f"exact={exact_overload(m, std_demand, capacity):.4%}")
for n in (100, 10_000, 1_000_000):
    d = rng.normal(mean_demand, std_demand, n)
    p = np.mean(d > capacity)
    print(f"N={n:>9,}: estimate={p:.4%}, error vs exact={abs(p - exact_overload(850, 120, 1000)):.4%}")

# ---------------------------------------------------------------
# PART 3: NEW CASES
# (a) Estimate pi from random points in the unit square
# (b) Project completion time: three sequential tasks with uncertain
#     durations (triangular: min, most likely, max in days):
#     Design (5, 8, 14), Build (10, 15, 30), Test (4, 6, 12).
#     Deadline = 32 days. Estimate the probability of finishing late.
# ---------------------------------------------------------------
print("\n=== PART 3: New cases ===")
for n in (1_000, 100_000, 5_000_000):
    pts = rng.random((n, 2))
    inside = np.sum(pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0)
    est = 4 * inside / n
    print(f"pi estimate with N={n:>9,}: {est:.5f} (error {abs(est - np.pi):.5f})")

M = 200_000
design = rng.triangular(5, 8, 14, M)
build = rng.triangular(10, 15, 30, M)
test = rng.triangular(4, 6, 12, M)
total = design + build + test
deadline = 32.0
print(f"Mean total duration: {total.mean():.2f} days (std {total.std():.2f})")
print(f"Probability of finishing after {deadline:.0f} days: {np.mean(total > deadline):.2%}")
print(f"90th percentile duration: {np.percentile(total, 90):.1f} days")
print(f"Deadline needed for 95 % confidence: {np.percentile(total, 95):.1f} days")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(indices + 1, running_probability[indices])
ax[0].axhline(exact_overload(850, 120, 1000), linestyle="--", linewidth=0.8, label="Exact")
ax[0].set_xlabel("Number of simulations"); ax[0].set_ylabel("Estimated overload probability")
ax[0].set_title("Monte Carlo estimate of server-overload risk"); ax[0].legend(); ax[0].grid(True)
ax[1].hist(total, bins=60, density=True)
ax[1].axvline(deadline, linestyle="--", color="red", label=f"Deadline {deadline:.0f} d")
ax[1].set_xlabel("Total project duration (days)"); ax[1].set_ylabel("Density")
ax[1].set_title("Project duration distribution"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_20_monte_carlo.png"), dpi=150)
plt.show()
