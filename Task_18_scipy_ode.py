"""Task 18: Numerical Solution of ODEs Using SciPy
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (RC charging circuit)
Part 2: Parameter change (capacitance, discharge case)
Part 3: New real-world case (SIR epidemic model)
"""
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_rc(Vs, R, C, V0, t_end, n=200):
    def rc_circuit(t, V):
        return [(Vs - V[0]) / (R * C)]
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(rc_circuit, (0, t_end), [V0], t_eval=t_eval, rtol=1e-9, atol=1e-11)
    exact = Vs + (V0 - Vs) * np.exp(-sol.t / (R * C))
    return sol, exact


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM  (Vs=5 V, R=1000, C=0.001 F -> tau = 1 s)
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
solution, exact = run_rc(5.0, 1000.0, 0.001, 0.0, 5.0)
print("Solver success:", solution.success)
print(f"Capacitor voltage at 5 s: {solution.y[0, -1]:.4f} V")
print(f"Maximum error against exact solution: {np.max(np.abs(solution.y[0] - exact)):.3e} V")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) C = 0.0005 F (time constant halves)  (b) discharge: Vs=0, V0=5 V
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
tau_orig = 1000.0 * 0.001
sol_c, _ = run_rc(5.0, 1000.0, 0.0005, 0.0, 5.0)
print(f"C=0.0005 F (tau = 0.5 s): V(1 s) = {np.interp(1.0, sol_c.t, sol_c.y[0]):.4f} V "
      f"(original C: V(1 s) = {np.interp(1.0, solution.t, solution.y[0]):.4f} V)")
print(f"Original tau = {tau_orig:.2f} s: V(tau) = {5 * (1 - np.exp(-1)):.4f} V (63.2 % of 5 V)")
sol_d, ex_d = run_rc(0.0, 1000.0, 0.001, 5.0, 5.0)
print(f"Discharge: V(1 s) = {np.interp(1.0, sol_d.t, sol_d.y[0]):.4f} V, "
      f"V(5 s) = {sol_d.y[0, -1]:.4f} V")

# ---------------------------------------------------------------
# PART 3: NEW CASE - SIR epidemic model
# dS/dt = -beta S I / N ; dI/dt = beta S I / N - gamma I ; dR/dt = gamma I
# Assumptions: school of N = 1000 students, 1 initial infection,
# beta = 0.5 /day (contacts), gamma = 0.2 /day (5-day infectious period)
# so R0 = beta/gamma = 2.5. Simulate 120 days.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - SIR epidemic ===")
N = 1000.0


def sir(t, y, beta, gamma):
    S, I, Rr = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


def run_sir(beta, gamma=0.2, days=120):
    return solve_ivp(sir, (0, days), [N - 1, 1, 0], args=(beta, gamma),
                     t_eval=np.linspace(0, days, 600), rtol=1e-8, atol=1e-10)


base = run_sir(0.5)
peak_i = base.y[1].max()
print(f"R0 = {0.5 / 0.2:.2f}")
print(f"Peak infected: {peak_i:.0f} students on day {base.t[base.y[1].argmax()]:.1f}")
print(f"Total ever infected (final R): {base.y[2, -1]:.0f} of {N:.0f}")
dist = run_sir(0.25)
print(f"With distancing (beta=0.25, R0={0.25 / 0.2:.2f}): peak = {dist.y[1].max():.0f}, "
      f"total infected = {dist.y[2, -1]:.0f}")
print("Check S+I+R conserved:", np.allclose(base.y.sum(axis=0), N))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(solution.t, solution.y[0], label="solve_ivp")
ax[0].plot(solution.t, exact, linestyle="--", label="Exact")
ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Capacitor voltage (V)")
ax[0].set_title("RC circuit charging"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(base.t, base.y[0], label="Susceptible")
ax[1].plot(base.t, base.y[1], label="Infected")
ax[1].plot(base.t, base.y[2], label="Recovered")
ax[1].plot(dist.t, dist.y[1], linestyle="--", label="Infected (distancing)")
ax[1].set_xlabel("Day"); ax[1].set_ylabel("Students")
ax[1].set_title("SIR epidemic model"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_18_ode.png"), dpi=150)
plt.show()
