"""Task 06: Solution of Systems of Linear Equations
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (two server types)
Part 2: Parameter change (different observed totals; singular case)
Part 3: New real-world case (three-loop electrical circuit)
"""
import numpy as np

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# x = number of Type A servers, y = number of Type B servers
# Type A: 4 CPU, 8 GB. Type B: 2 CPU, 12 GB. Totals: 20 CPU, 72 GB
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
A = np.array([
    [4.0, 2.0],
    [8.0, 12.0]
])
b = np.array([20.0, 72.0])

solution = np.linalg.solve(A, b)
residual = b - A @ solution

print(f"Type A servers: {solution[0]:.0f}")
print(f"Type B servers: {solution[1]:.0f}")
print("Verification A @ solution:", A @ solution)
print("Residual:", residual)
print("Residual norm:", np.linalg.norm(residual))

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) New totals: 32 CPU units and 112 GB memory
# (b) Type B now uses 1 CPU and 6 GB (proportional to Type A): the two
#     rows become dependent and the system has no unique solution.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
b2 = np.array([32.0, 112.0])
sol2 = np.linalg.solve(A, b2)
print(f"(a) Totals 32 CPU / 112 GB -> Type A = {sol2[0]:.2f}, Type B = {sol2[1]:.2f}")

A_sing = np.array([[4.0, 2.0], [8.0, 4.0]])
print("(b) Determinant of dependent system:", np.linalg.det(A_sing))
try:
    np.linalg.solve(A_sing, np.array([20.0, 40.0]))
except np.linalg.LinAlgError as err:
    print("    Solver reports:", err)

# ---------------------------------------------------------------
# PART 3: NEW CASE - three-loop DC circuit (mesh-current analysis)
# Assumptions: R1=10, R2=20, R3=30, R4=40, R5=50 ohms; source V1=12 V in
# loop 1 and V2=5 V (opposing) in loop 3. Unknowns: mesh currents I1,I2,I3.
#   (R1+R2) I1 - R2 I2             = V1
#   -R2 I1 + (R2+R3+R4) I2 - R4 I3 = 0
#   -R4 I2 + (R4+R5) I3            = -V2
# ---------------------------------------------------------------
print("\n=== PART 3: New case - mesh currents in a circuit ===")
R1, R2, R3, R4, R5 = 10.0, 20.0, 30.0, 40.0, 50.0
V1, V2 = 12.0, 5.0
Ac = np.array([
    [R1 + R2, -R2, 0.0],
    [-R2, R2 + R3 + R4, -R4],
    [0.0, -R4, R4 + R5]
])
bc = np.array([V1, 0.0, -V2])
currents = np.linalg.solve(Ac, bc)
for name, i in zip(("I1", "I2", "I3"), currents):
    print(f"{name} = {i * 1000:.3f} mA")
res = bc - Ac @ currents
print("Residual norm:", np.linalg.norm(res))
print("Condition number of matrix:", round(np.linalg.cond(Ac), 2))
print(f"Current through R2 (I1-I2): {(currents[0] - currents[1]) * 1000:.3f} mA")
