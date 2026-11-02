"""Task 10: Root Finding - Bisection and Fixed-Point Methods
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (heat-balance equilibrium temperature)
Part 2: Parameter change (bracket, tolerance, fixed-point start)
Part 3: New real-world case (business break-even quantity)
"""
import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Heat-balance model: f(T) = 0 means heat gain equals heat loss.
def heat_balance(T):
    return T - 25 - 10 * np.exp(-T / 20)


# Fixed-point form: T = 25 + 10*exp(-T/20)
def g(T):
    return 25 + 10 * np.exp(-T / 20)


def fixed_point(g, x0, tol=1e-10, max_iter=100):
    x = x0
    hist = [x]
    for _ in range(max_iter):
        x_new = g(x)
        hist.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return hist


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
solution = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=1e-10)
history = fixed_point(g, 30.0)

print(f"Bisection equilibrium temperature: {solution.root:.6f} C")
print(f"Fixed-point equilibrium temperature: {history[-1]:.6f} C")
print("Fixed-point iterations:", len(history) - 1)
print("Bisection iterations:", solution.iterations)

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) tolerance 1e-3 vs 1e-10   (b) narrower bracket [24, 30]
# (c) fixed-point start 100 C instead of 30 C
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
for tol in (1e-3, 1e-6, 1e-10):
    s = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=tol)
    print(f"xtol={tol:.0e}: root={s.root:.8f}, iterations={s.iterations}")
s2 = root_scalar(heat_balance, bracket=[24, 30], method="bisect", xtol=1e-10)
print(f"Bracket [24,30]: root={s2.root:.8f}, iterations={s2.iterations}")
h2 = fixed_point(g, 100.0)
print(f"Fixed point from 100 C: {h2[-1]:.8f} C in {len(h2) - 1} iterations")
try:
    root_scalar(heat_balance, bracket=[30, 40], method="bisect")
except ValueError as e:
    print("Bracket [30,40] has no sign change ->", e)

# ---------------------------------------------------------------
# PART 3: NEW CASE - break-even point for a small business
# Assumptions: price per unit = 50; variable cost = 20 per unit; fixed cost
# = 2000; extra congestion cost 0.01*q^2 at high volume.
# Profit(q) = 50q - (2000 + 20q + 0.01 q^2). Break-even where Profit = 0.
# There are two break-even points (low volume and capacity limit).
# ---------------------------------------------------------------
print("\n=== PART 3: New case - break-even quantities ===")


def profit(q):
    return 50 * q - (2000 + 20 * q + 0.01 * q ** 2)


low = root_scalar(profit, bracket=[10, 500], method="bisect", xtol=1e-8)
high = root_scalar(profit, bracket=[500, 5000], method="bisect", xtol=1e-8)
print(f"Lower break-even quantity : {low.root:.2f} units")
print(f"Upper break-even quantity : {high.root:.2f} units")
q_best = 30 / (2 * 0.01)
print(f"Most profitable quantity (vertex): {q_best:.0f} units, profit = {profit(q_best):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
temps = np.linspace(20, 40, 300)
ax[0].plot(temps, heat_balance(temps))
ax[0].axhline(0, linewidth=0.8)
ax[0].axvline(solution.root, linestyle="--")
ax[0].set_xlabel("Temperature (C)"); ax[0].set_ylabel("Heat-balance function")
ax[0].set_title("Equilibrium temperature"); ax[0].grid(True)
q = np.linspace(0, 3200, 400)
ax[1].plot(q, profit(q))
ax[1].axhline(0, linewidth=0.8)
ax[1].axvline(low.root, linestyle="--"); ax[1].axvline(high.root, linestyle="--")
ax[1].set_xlabel("Quantity sold"); ax[1].set_ylabel("Profit")
ax[1].set_title("Break-even points"); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_10_roots.png"), dpi=150)
plt.show()
