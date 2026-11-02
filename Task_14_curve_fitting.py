"""Task 14: Curve Fitting and Approximation
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (server response time, quadratic fit)
Part 2: Parameter change (linear vs quadratic vs cubic model)
Part 3: New real-world case (logistic growth of mobile-money users)
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
users = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
response_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)


def model(x, a, b, c):
    return a * x ** 2 + b * x + c


parameters, covariance = curve_fit(model, users, response_ms)
a, b, c = parameters
predicted = model(users, a, b, c)
rmse = np.sqrt(np.mean((response_ms - predicted) ** 2))

print(f"Fitted model: y = {a:.6f}x^2 + {b:.4f}x + {c:.2f}")
print(f"RMSE: {rmse:.2f} ms")
print(f"Predicted response at 400 users: {model(400, a, b, c):.2f} ms")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - compare polynomial degrees 1, 2, 3
# and see how the 400-user prediction changes.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - model degree ===")
for deg in (1, 2, 3):
    coef = np.polyfit(users, response_ms, deg)
    pred = np.polyval(coef, users)
    r = np.sqrt(np.mean((response_ms - pred) ** 2))
    print(f"Degree {deg}: RMSE = {r:.2f} ms, prediction at 400 users = "
          f"{np.polyval(coef, 400):.1f} ms")
print("Parameter uncertainties (std dev) of quadratic fit:",
      np.round(np.sqrt(np.diag(covariance)), 6))

# ---------------------------------------------------------------
# PART 3: NEW CASE - logistic growth of mobile-money subscribers
# Assumptions: yearly subscriber counts (millions) for 10 years, generated
# from a logistic curve (capacity 40 M, rate 0.8, midpoint year 5) plus
# small noise (seed 7). Model: N(t) = K / (1 + exp(-r (t - t0))).
# ---------------------------------------------------------------
print("\n=== PART 3: New case - logistic growth ===")
rng = np.random.default_rng(7)
year = np.arange(0, 10, dtype=float)


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


subs = logistic(year, 40.0, 0.8, 5.0) + rng.normal(0, 0.8, year.size)
popt, pcov = curve_fit(logistic, year, subs, p0=[30, 0.5, 4])
K, r, t0 = popt
pred_l = logistic(year, *popt)
rmse_l = np.sqrt(np.mean((subs - pred_l) ** 2))
print(f"Fitted capacity K = {K:.2f} M, growth rate r = {r:.3f}, midpoint t0 = {t0:.2f}")
print(f"RMSE: {rmse_l:.2f} M")
print(f"Forecast for year 12: {logistic(12, *popt):.2f} M")
print(f"Residuals (M): {np.round(subs - pred_l, 2)}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x_plot = np.linspace(users.min(), 400, 300)
ax[0].scatter(users, response_ms, label="Measurements")
ax[0].plot(x_plot, model(x_plot, a, b, c), label="Fitted curve")
ax[0].set_xlabel("Concurrent users"); ax[0].set_ylabel("Response time (ms)")
ax[0].set_title("Server response-time curve fitting"); ax[0].legend(); ax[0].grid(True)
t_plot = np.linspace(0, 14, 300)
ax[1].scatter(year, subs, label="Observed")
ax[1].plot(t_plot, logistic(t_plot, *popt), label="Logistic fit")
ax[1].set_xlabel("Year"); ax[1].set_ylabel("Subscribers (millions)")
ax[1].set_title("Logistic growth fit"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_14_curve_fit.png"), dpi=150)
plt.show()
