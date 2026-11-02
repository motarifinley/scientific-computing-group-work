"""Task 16: Numerical Integration
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (smart-meter energy)
Part 2: Parameter change (fewer readings; accuracy test on a known integral)
Part 3: New real-world case (total water volume from flow-rate data)
"""
import numpy as np
from scipy.integrate import simpson, quad

# NumPy 2.x renamed trapz -> trapezoid
trap = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

energy_trapezoid = trap(power_kw, time_hours)
energy_simpson = simpson(power_kw, x=time_hours)

print(f"Energy using trapezoidal rule: {energy_trapezoid:.3f} kWh")
print(f"Energy using Simpson's rule: {energy_simpson:.3f} kWh")
print(f"Difference between estimates: {abs(energy_simpson - energy_trapezoid):.3f} kWh")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) keep only every second reading (4-hourly: 0,4,8,12)
# (b) accuracy test: integral of sin(x) from 0 to pi = 2 exactly
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
t4, p4 = time_hours[::2], power_kw[::2]
print(f"4-hourly data -> trapezoid {trap(p4, t4):.3f} kWh, Simpson {simpson(p4, x=t4):.3f} kWh")

for n in (3, 5, 9, 33):
    x = np.linspace(0, np.pi, n)
    y = np.sin(x)
    et = abs(trap(y, x) - 2)
    es = abs(simpson(y, x=x) - 2)
    print(f"n={n:>3} points: trapezoid error={et:.2e}, Simpson error={es:.2e}")
val, err = quad(np.sin, 0, np.pi)
print(f"Adaptive quad: {val:.12f} (estimated error {err:.1e})")

# ---------------------------------------------------------------
# PART 3: NEW CASE - total river/pipe water volume
# Assumptions: a flow meter records flow rate (m^3/s) every 10 minutes for
# 2 hours (13 readings). Volume = integral of flow rate over time (convert
# minutes to seconds). Also total distance from velocity as a second use.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - water volume from flow rate ===")
minutes = np.arange(0, 130, 10, dtype=float)
flow = np.array([2.0, 2.4, 3.1, 4.0, 4.8, 5.2, 5.0, 4.4, 3.7, 3.0, 2.6, 2.2, 2.0])
seconds = minutes * 60.0
vol_trap = trap(flow, seconds)
vol_simp = simpson(flow, x=seconds)
print(f"Volume (trapezoid): {vol_trap:,.0f} m^3")
print(f"Volume (Simpson)  : {vol_simp:,.0f} m^3")
print(f"Average flow rate : {vol_simp / seconds[-1]:.3f} m^3/s")
print(f"Difference        : {abs(vol_trap - vol_simp):,.0f} m^3 "
      f"({100 * abs(vol_trap - vol_simp) / vol_simp:.2f} %)")
