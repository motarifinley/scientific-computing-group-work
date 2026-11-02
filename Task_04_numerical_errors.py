"""Task 04: Floating-Point Arithmetic and Numerical Errors
BIT 2118 Scientific Computing - Group Work

Part 1: Original program
Part 2: Parameter change (approximation of sqrt(2), float32 vs float64)
Part 3: New real-world case (banking: summing cents)
"""
import math
from decimal import Decimal
import numpy as np

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
calculated = 0.1 + 0.2
expected = 0.3

print("0.1 + 0.2 =", calculated)
print("Exact comparison:", calculated == expected)
print("Tolerance-based comparison:", np.isclose(calculated, expected))

true_value = np.sqrt(2)
approx_value = 1.414
absolute_error = abs(true_value - approx_value)
relative_error = absolute_error / abs(true_value)
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")

# Numerically unstable vs stable formulation
for x in [1e-4, 1e-8, 1e-12]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) a better approximation 1.41421 instead of 1.414
# (b) smaller x values (1e-14, 1e-16) in the unstable formula
# (c) float32 instead of float64 when adding 0.1 ten thousand times
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
for approx in (1.414, 1.41421, 1.4142135):
    ae = abs(np.sqrt(2) - approx)
    print(f"approx={approx}: abs error={ae:.3e}, relative={ae / np.sqrt(2):.3e}")

for x in [1e-14, 1e-16]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")

total32 = np.float32(0.0)
total64 = np.float64(0.0)
for _ in range(10_000):
    total32 += np.float32(0.1)
    total64 += np.float64(0.1)
print(f"Sum of 0.1 x 10,000 -> float32: {total32:.6f}, float64: {total64:.10f}, "
      f"true: 1000")

# ---------------------------------------------------------------
# PART 3: NEW CASE - banking: adding one million payments of 0.01
# Assumptions: each payment is exactly 0.01 currency units (1 cent).
# True total = 10,000.00. Compare naive float sum, math.fsum and Decimal.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - summing one million cents ===")
n = 1_000_000
naive = 0.0
for _ in range(n):
    naive += 0.01
exact_fsum = math.fsum([0.01] * n)
exact_decimal = sum(Decimal("0.01") for _ in range(n))

print(f"Naive float sum : {naive:.10f}")
print(f"math.fsum       : {exact_fsum:.10f}")
print(f"Decimal sum     : {exact_decimal}")
print(f"Error of naive sum: {abs(naive - 10000):.3e}")
print("Is naive sum == 10000?", naive == 10000, "| isclose:", np.isclose(naive, 10000))
