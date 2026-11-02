"""Task 07: Gaussian Elimination and LU Decomposition
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (LU reused for several measurement vectors)
Part 2: Parameter change (many right-hand sides, timing LU vs repeated solve)
Part 3: New real-world case (spring-mass structure under changing loads)
"""
import timeit
import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
A = np.array([
    [4.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 3.0]
])
measurement_sets = [
    np.array([15.0, 10.0, 10.0]),
    np.array([16.0, 11.0, 9.0]),
    np.array([14.0, 12.0, 11.0])
]

lu, piv = lu_factor(A)
for i, b in enumerate(measurement_sets, start=1):
    x = lu_solve((lu, piv), b)
    residual = b - A @ x
    print(f"Measurement set {i}")
    print("Solution:", x)
    print("Residual norm:", np.linalg.norm(residual))
    print()

print("Direct solution for first set:", np.linalg.solve(A, measurement_sets[0]))

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - 500x500 matrix, 200 right-hand sides
# Compare: factor once + 200 cheap solves vs 200 full solves.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - larger system, many b vectors ===")
n, m = 500, 200
rng = np.random.default_rng(1)
big = rng.normal(size=(n, n)) + n * np.eye(n)  # well conditioned
bs = rng.normal(size=(m, n))


def repeated_solve():
    return [np.linalg.solve(big, b) for b in bs]


def factor_once():
    lu_b, piv_b = lu_factor(big)
    return [lu_solve((lu_b, piv_b), b) for b in bs]


t_rep = timeit.timeit(repeated_solve, number=1)
t_lu = timeit.timeit(factor_once, number=1)
print(f"200 full solves   : {t_rep:.4f} s")
print(f"LU once + 200 solves: {t_lu:.4f} s")
print(f"Speed-up: {t_rep / t_lu:.1f}x")
x_a = np.linalg.solve(big, bs[0])
lu_b, piv_b = lu_factor(big)
x_b = lu_solve((lu_b, piv_b), bs[0])
print("Both methods agree:", np.allclose(x_a, x_b))

# ---------------------------------------------------------------
# PART 3: NEW CASE - four-mass spring chain under different load cases
# Assumptions: stiffness matrix K (N/mm) is fixed for the structure; each
# load case F (N) is a different set of forces on the four joints. Solve
# K u = F for displacements u (mm). K is factored once.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - structural displacements ===")
K = np.array([
    [ 200., -100.,    0.,    0.],
    [-100.,  200., -100.,    0.],
    [   0., -100.,  200., -100.],
    [   0.,    0., -100.,  100.]
])
loads = {
    "Load case 1 (light)":  np.array([0.0, 0.0, 0.0, 50.0]),
    "Load case 2 (medium)": np.array([10.0, 10.0, 10.0, 100.0]),
    "Load case 3 (uneven)": np.array([200.0, 0.0, 0.0, 0.0]),
}
lu_k, piv_k = lu_factor(K)
for name, F in loads.items():
    u = lu_solve((lu_k, piv_k), F)
    print(name, "-> displacements (mm):", np.round(u, 4),
          "| residual:", f"{np.linalg.norm(F - K @ u):.1e}")
print("Determinant of K:", round(np.linalg.det(K), 2))
