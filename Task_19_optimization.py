"""Task 19: Numerical Optimization
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (cheapest cloud server allocation)
Part 2: Parameter change (cost and capacity requirement)
Part 3: New real-world case (delivery fleet with two constraints)
"""
import numpy as np
from scipy.optimize import minimize


def solve_cloud(cost_a=12, cost_b=18, cap_a=100, cap_b=170, required=1000, x0=(5, 3)):
    cons = {"type": "ineq", "fun": lambda z: cap_a * z[0] + cap_b * z[1] - required}
    res = minimize(lambda z: cost_a * z[0] + cost_b * z[1], x0=list(x0),
                   method="SLSQP", bounds=[(0, None), (0, None)], constraints=cons)
    best = None
    for a in range(0, 40):
        for b in range(0, 40):
            if cap_a * a + cap_b * b >= required:
                c = cost_a * a + cost_b * b
                if best is None or c < best[0]:
                    best = (c, a, b, cap_a * a + cap_b * b)
    return res, best


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
solution, best_integer = solve_cloud()
x, y = solution.x
print("Optimization success:", solution.success)
print(f"Type A allocation: {x:.3f}")
print(f"Type B allocation: {y:.3f}")
print(f"Total capacity: {100 * x + 170 * y:.2f}")
print(f"Minimum planning cost: {solution.fun:.2f}")
print("Best nearby integer solution (cost, A, B, capacity):", best_integer)

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) Type B cost rises 18 -> 25 (A becomes cheaper per unit capacity)
# (b) required capacity 1000 -> 1500
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
s_a, i_a = solve_cloud(cost_b=25)
print(f"(a) Cost B=25: A={s_a.x[0]:.3f}, B={s_a.x[1]:.3f}, cost={s_a.fun:.2f}, integer best={i_a}")
s_b, i_b = solve_cloud(required=1500)
print(f"(b) Capacity 1500: A={s_b.x[0]:.3f}, B={s_b.x[1]:.3f}, cost={s_b.fun:.2f}, integer best={i_b}")
print("Cost per unit capacity: A =", 12 / 100, " B =", round(18 / 170, 4),
      "-> the cheaper ratio is used exclusively")

# ---------------------------------------------------------------
# PART 3: NEW CASE - delivery fleet planning
# Assumptions: vans cost 40 per day and carry 30 parcels / 400 kg;
# lorries cost 90 per day and carry 80 parcels / 1200 kg.
# The day requires at least 500 parcels and 8000 kg moved.
# Minimise cost with two inequality constraints, then find whole vehicles.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - delivery fleet ===")
cons = [
    {"type": "ineq", "fun": lambda z: 30 * z[0] + 80 * z[1] - 500},
    {"type": "ineq", "fun": lambda z: 400 * z[0] + 1200 * z[1] - 8000},
]
res = minimize(lambda z: 40 * z[0] + 90 * z[1], x0=[5, 5], method="SLSQP",
               bounds=[(0, None), (0, None)], constraints=cons)
print("Success:", res.success)
print(f"Continuous: vans={res.x[0]:.3f}, lorries={res.x[1]:.3f}, cost={res.fun:.2f}")
best = None
for v in range(0, 40):
    for l in range(0, 40):
        if 30 * v + 80 * l >= 500 and 400 * v + 1200 * l >= 8000:
            c = 40 * v + 90 * l
            if best is None or c < best[0]:
                best = (c, v, l, 30 * v + 80 * l, 400 * v + 1200 * l)
print("Best whole-vehicle plan (cost, vans, lorries, parcels, kg):", best)
print("Active constraints at continuous optimum:",
      "parcels" if abs(30 * res.x[0] + 80 * res.x[1] - 500) < 1e-3 else "-",
      "weight" if abs(400 * res.x[0] + 1200 * res.x[1] - 8000) < 1e-3 else "-")
