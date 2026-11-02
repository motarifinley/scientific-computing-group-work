"""Task 12: Polynomial and Nonlinear Equation Solving
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (circle and parabola intersections)
Part 2: Parameter change (different radius and starting guesses)
Part 3: New real-world case (projectile motion: polynomial roots)
"""
import numpy as np
from scipy.optimize import root


def make_equations(radius):
    def equations(z):
        x, y = z
        return [
            x ** 2 + y ** 2 - radius ** 2,
            y - (x ** 2 - 1)
        ]
    return equations


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM (radius = 2)
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
equations = make_equations(2.0)
for guess in ([1.0, 1.0], [-1.0, 1.0]):
    solution = root(equations, guess)
    print("Initial guess:", guess)
    print("Intersection:", solution.x)
    print("Residuals:", equations(solution.x))
    print("Converged:", solution.success)
    print()

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) radius 1 instead of 2   (b) a poor initial guess (0, -5)
# ---------------------------------------------------------------
print("=== PART 2: Parameter change ===")
eq1 = make_equations(1.0)
for guess in ([1.0, 0.0], [-1.0, 0.0]):
    s = root(eq1, guess)
    print(f"Radius 1, guess {guess}: x={s.x}, converged={s.success}, "
          f"residual norm={np.linalg.norm(eq1(s.x)):.2e}")
s_bad = root(equations, [0.0, -5.0])
print("Radius 2, guess [0,-5]: x =", s_bad.x, "converged =", s_bad.success,
      "| residual norm =", f"{np.linalg.norm(equations(s_bad.x)):.3f}")
print("-> a converged flag of False (or a large residual) means NO true intersection was found.")

# Polynomial from the concept section: x^2 - 5x + 6 = 0
print("Roots of x^2 - 5x + 6:", np.roots([1, -5, 6]))

# ---------------------------------------------------------------
# PART 3: NEW CASE - projectile launched from a 2 m platform
# Assumptions: initial vertical speed 20 m/s, g = 9.81 m/s^2, no air drag.
#   h(t) = -4.905 t^2 + 20 t + 2
# (a) time when the ball hits the ground (h = 0)
# (b) times when the ball is at a 15 m height (h = 15)
# ---------------------------------------------------------------
print("\n=== PART 3: New case - projectile ===")
g, v0, h0 = 9.81, 20.0, 2.0
coeffs = [-g / 2, v0, h0]
roots_ground = np.roots(coeffs)
print("Roots of h(t)=0:", np.round(roots_ground, 4))
t_land = roots_ground[roots_ground > 0][0]
print(f"Ball lands after {t_land:.3f} s (negative root is not physical)")

coeffs15 = [-g / 2, v0, h0 - 15.0]
r15 = np.roots(coeffs15)
print(f"Height 15 m reached at t = {np.round(np.sort(r15), 4)} s (up and down)")
t_peak = v0 / g
print(f"Peak at t = {t_peak:.3f} s, maximum height = {h0 + v0 * t_peak - g / 2 * t_peak ** 2:.3f} m")
# Verify with nonlinear solver
sol = root(lambda t: -g / 2 * t ** 2 + v0 * t + h0, 4.0)
print(f"Check with scipy.optimize.root: t = {sol.x[0]:.4f} s, converged={sol.success}")
