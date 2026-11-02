"""Task 13: Interpolation
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (hourly temperature -> 10-minute estimates)
Part 2: Parameter change (different target time and a sparser sensor)
Part 3: New real-world case (filling missing river-level readings)
"""
import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

ten_minute_times = np.arange(8, 14 + 1 / 6, 1 / 6)
linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5  # 10:30
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} C")
print(f"Cubic estimate at 10:30: {float(cubic(target_time)):.2f} C")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) target time 13:30 near the peak   (b) sensor now reports every
# 2 hours only (drop odd hours) and we test the error at removed hours.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
t2 = 13.5
print(f"At 13:30 -> linear: {float(linear(t2)):.3f} C, cubic: {float(cubic(t2)):.3f} C")

h_sparse = hours[::2]
T_sparse = temperature[::2]
lin_s = interp1d(h_sparse, T_sparse, kind="linear")
cub_s = CubicSpline(h_sparse, T_sparse)
for h in hours[1::2]:
    true = temperature[hours == h][0]
    print(f"Hour {h:.0f}: true={true:.2f}, linear={float(lin_s(h)):.2f} "
          f"(err {abs(float(lin_s(h)) - true):.2f}), cubic={float(cub_s(h)):.2f} "
          f"(err {abs(float(cub_s(h)) - true):.2f})")
try:
    linear(15.0)
except ValueError as e:
    print("Extrapolation to 15:00 refused by interp1d ->", str(e)[:60], "...")

# ---------------------------------------------------------------
# PART 3: NEW CASE - missing river-level readings
# Assumptions: a gauge records water level (m) every hour, but the
# readings at hours 3, 4 and 9 are missing (NaN, sensor outage). Estimate
# them from the valid readings; flood warning level is 2.6 m.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - river-level gaps ===")
t = np.arange(0, 13, dtype=float)
level = np.array([1.80, 1.85, 1.95, np.nan, np.nan, 2.45, 2.60,
                  2.72, 2.78, np.nan, 2.70, 2.58, 2.45])
ok = ~np.isnan(level)
lin_r = interp1d(t[ok], level[ok], kind="linear")
cub_r = CubicSpline(t[ok], level[ok])
missing = t[~ok]
for m in missing:
    print(f"Hour {m:.0f}: linear = {float(lin_r(m)):.3f} m, cubic = {float(cub_r(m)):.3f} m")
filled = level.copy()
filled[~ok] = cub_r(missing)
print("Hours at/above flood warning (2.6 m):", t[filled >= 2.6])
fine = np.linspace(0, 12, 200)
print(f"Estimated peak level: {cub_r(fine).max():.3f} m at hour {fine[cub_r(fine).argmax()]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(hours, temperature, label="Hourly measurements")
ax[0].plot(ten_minute_times, linear_estimates, label="Linear")
ax[0].plot(ten_minute_times, cubic_estimates, label="Cubic spline")
ax[0].set_xlabel("Hour of day"); ax[0].set_ylabel("Temperature (C)")
ax[0].set_title("Temperature interpolation"); ax[0].legend(); ax[0].grid(True)
ax[1].scatter(t[ok], level[ok], label="Valid readings")
ax[1].scatter(missing, cub_r(missing), marker="x", color="red", label="Estimated (missing)")
ax[1].plot(fine, cub_r(fine), label="Cubic spline")
ax[1].axhline(2.6, linestyle="--", linewidth=0.8, label="Flood warning")
ax[1].set_xlabel("Hour"); ax[1].set_ylabel("Water level (m)")
ax[1].set_title("River-level gap filling"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_13_interpolation.png"), dpi=150)
plt.show()
