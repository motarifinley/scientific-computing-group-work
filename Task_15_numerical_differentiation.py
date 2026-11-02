"""Task 15: Numerical Differentiation
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (speed from GPS position)
Part 2: Parameter change (step size h, forward vs central difference)
Part 3: New real-world case (train acceleration from velocity data)
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
time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

speed = np.zeros_like(position)
speed[0] = (position[1] - position[0]) / (time[1] - time[0])
for i in range(1, len(time) - 1):
    speed[i] = (position[i + 1] - position[i - 1]) / (time[i + 1] - time[i - 1])
speed[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])

for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s, position={x:.1f}m, estimated speed={v:.2f}m/s")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - step size h on a function with a known
# derivative: f(x) = sin(x), f'(1) = cos(1). Compare forward & central.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - step size h ===")
x0 = 1.0
exact = np.cos(x0)
print(f"{'h':>8} {'forward err':>14} {'central err':>14}")
for h in (1.0, 0.1, 0.01, 0.001, 1e-6, 1e-10, 1e-14):
    fwd = (np.sin(x0 + h) - np.sin(x0)) / h
    cen = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)
    print(f"{h:8.0e} {abs(fwd - exact):14.3e} {abs(cen - exact):14.3e}")
print("-> error falls as h shrinks, then rises again (round-off) for tiny h.")

# ---------------------------------------------------------------
# PART 3: NEW CASE - train acceleration from velocity data
# Assumptions: speed (m/s) logged every 5 s as a train leaves a station,
# cruises, then brakes. Acceleration = dv/dt using central differences
# (np.gradient). Comfort limit: |a| <= 1.0 m/s^2.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - train acceleration ===")
t_tr = np.arange(0, 65, 5, dtype=float)
v_tr = np.array([0, 4.0, 9.0, 14.0, 18.0, 21.0, 22.0, 22.0,
                 22.0, 22.0, 18.0, 11.0, 4.0])
accel = np.gradient(v_tr, t_tr)
for ti, vi, ai in zip(t_tr, v_tr, accel):
    flag = "  <-- exceeds comfort limit" if abs(ai) > 1.0 else ""
    print(f"t={ti:4.0f}s  v={vi:5.1f} m/s  a={ai:6.3f} m/s^2{flag}")
print(f"Maximum acceleration : {accel.max():.3f} m/s^2")
print(f"Maximum deceleration : {accel.min():.3f} m/s^2")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(time, speed, marker="o")
ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Estimated speed (m/s)")
ax[0].set_title("Speed estimated from GPS position data"); ax[0].grid(True)
ax[1].plot(t_tr, accel, marker="o")
ax[1].axhline(1.0, linestyle="--", linewidth=0.8); ax[1].axhline(-1.0, linestyle="--", linewidth=0.8)
ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("Acceleration (m/s^2)")
ax[1].set_title("Train acceleration from velocity"); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_15_differentiation.png"), dpi=150)
plt.show()
