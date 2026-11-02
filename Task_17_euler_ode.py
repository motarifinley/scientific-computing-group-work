"""Task 17: Ordinary Differential Equations - Euler Method
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (Newton's law of cooling)
Part 2: Parameter change (step size h)
Part 3: New real-world case (logistic population growth)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def euler_cooling(h, T_initial=90.0, T_ambient=25.0, k=0.12, end_time=30.0):
    time = np.arange(0, end_time + h / 2, h)
    temp = np.zeros_like(time)
    temp[0] = T_initial
    for i in range(len(time) - 1):
        temp[i + 1] = temp[i] + h * (-k * (temp[i] - T_ambient))
    exact = T_ambient + (T_initial - T_ambient) * np.exp(-k * time)
    return time, temp, exact


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM  (h = 1 minute)
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
time, temperature, exact = euler_cooling(1.0)
print(f"Euler temperature after 30 min: {temperature[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error: {abs(temperature[-1] - exact[-1]):.4f} C")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - step size h
# Stability requires |1 - k h| < 1, i.e. h < 2/k = 16.7 min here.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - step size ===")
for h in (5.0, 1.0, 0.1, 0.01):
    t_, T_, ex_ = euler_cooling(h)
    print(f"h={h:>5}: Euler={T_[-1]:8.4f} C, exact={ex_[-1]:.4f} C, "
          f"error={abs(T_[-1] - ex_[-1]):.5f} C, steps={len(t_) - 1}")
t_u, T_u, ex_u = euler_cooling(20.0, end_time=100.0)
print(f"h=20 (unstable, > 16.7): Euler values start {np.round(T_u[:4], 1)} "
      f"and oscillate while exact decays to {ex_u[-1]:.1f}")

# ---------------------------------------------------------------
# PART 3: NEW CASE - logistic population growth
# dP/dt = r P (1 - P/K). Assumptions: a fish population in a lake,
# P(0) = 500, growth rate r = 0.5 per year, carrying capacity K = 10,000,
# simulated for 20 years. Exact: P(t) = K / (1 + ((K-P0)/P0) e^{-r t}).
# ---------------------------------------------------------------
print("\n=== PART 3: New case - logistic population ===")
r, K, P0 = 0.5, 10_000.0, 500.0
h_p = 0.5
tp = np.arange(0, 20 + h_p / 2, h_p)
P = np.zeros_like(tp)
P[0] = P0
for i in range(len(tp) - 1):
    P[i + 1] = P[i] + h_p * r * P[i] * (1 - P[i] / K)
P_exact = K / (1 + ((K - P0) / P0) * np.exp(-r * tp))
print(f"Euler population after 20 yr: {P[-1]:.1f}")
print(f"Exact population after 20 yr: {P_exact[-1]:.1f}")
print(f"Maximum absolute error over run: {np.max(np.abs(P - P_exact)):.1f} fish")
half = tp[np.argmax(P >= K / 2)]
print(f"Population first exceeds K/2 (5000) at about t = {half:.1f} years")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(time, temperature, marker="o", label="Euler")
ax[0].plot(time, exact, label="Exact solution")
ax[0].set_xlabel("Time (minutes)"); ax[0].set_ylabel("Temperature (C)")
ax[0].set_title("Cooling of a hot object"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(tp, P, marker="o", markersize=3, label="Euler")
ax[1].plot(tp, P_exact, label="Exact")
ax[1].axhline(K, linestyle="--", linewidth=0.8)
ax[1].set_xlabel("Time (years)"); ax[1].set_ylabel("Population")
ax[1].set_title("Logistic population growth"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_17_euler.png"), dpi=150)
plt.show()
