"""Task 08: Iterative Methods for Linear Systems (Jacobi, Gauss-Seidel)
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (5 grid-point temperature model)
Part 2: Parameter change (tolerance and boundary temperatures)
Part 3: New real-world case (heated rod with 20 interior points)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def jacobi(A, b, tol=1e-8, max_iter=500):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=500):
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (
                b[i]
                - A[i, :i] @ x[:i]
                - A[i, i + 1:] @ old[i + 1:]
            ) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors
    return x, errors


def tridiag(n):
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
A = tridiag(5)
b = np.array([100., 0., 0., 0., 20.])

x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

print("Jacobi temperatures:", x_j)
print("Gauss-Seidel temperatures:", x_gs)
print("Direct solution:", x_exact)
print(f"Iterations -> Jacobi: {len(err_j)}, Gauss-Seidel: {len(err_gs)}")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) looser tolerance 1e-3  (b) boundaries 200 C and 50 C
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
for tol in (1e-3, 1e-8, 1e-12):
    _, ej = jacobi(A, b, tol=tol)
    _, eg = gauss_seidel(A, b, tol=tol)
    print(f"tol={tol:.0e}: Jacobi iters={len(ej)}, Gauss-Seidel iters={len(eg)}")

b_new = np.array([200., 0., 0., 0., 50.])
x_new, e_new = gauss_seidel(A, b_new)
print("New boundaries 200/50 C -> temperatures:", np.round(x_new, 4),
      f"({len(e_new)} iterations)")

# ---------------------------------------------------------------
# PART 3: NEW CASE - heated metal rod, 20 interior points
# Assumptions: steady state, no internal heat source, left end held at
# 80 C, right end at 20 C. Equation: -T[i-1] + 2T[i] - T[i+1] = 0 with the
# boundary values moved to the right-hand side.
# Exact answer is a straight line from 80 C to 20 C.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - heated rod ===")
n = 20
A_rod = tridiag(n)
b_rod = np.zeros(n)
b_rod[0] = 80.0
b_rod[-1] = 20.0
x_rod_j, e_rod_j = jacobi(A_rod, b_rod, tol=1e-6, max_iter=5000)
x_rod_gs, e_rod_gs = gauss_seidel(A_rod, b_rod, tol=1e-6, max_iter=5000)
exact_line = 80.0 + (20.0 - 80.0) * np.arange(1, n + 1) / (n + 1)
print(f"Jacobi iterations: {len(e_rod_j)}, Gauss-Seidel iterations: {len(e_rod_gs)}")
print("Max error vs straight-line solution (Gauss-Seidel):",
      f"{np.max(np.abs(x_rod_gs - exact_line)):.2e}")
print("Temperatures along rod (GS):", np.round(x_rod_gs, 2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(err_j, label="Jacobi")
ax[0].semilogy(err_gs, label="Gauss-Seidel")
ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("Maximum change")
ax[0].set_title("Convergence of iterative temperature solution")
ax[0].legend(); ax[0].grid(True)
ax[1].plot(range(1, n + 1), x_rod_gs, "o-", label="Gauss-Seidel")
ax[1].plot(range(1, n + 1), exact_line, "--", label="Exact line")
ax[1].set_xlabel("Grid point"); ax[1].set_ylabel("Temperature (C)")
ax[1].set_title("Heated rod temperature profile")
ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_08_convergence.png"), dpi=150)
plt.show()
