"""Task 11: Root Finding - Newton-Raphson and Secant Methods
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (IRR of an investment)
Part 2: Parameter change (cash flows and starting guesses)
Part 3: New real-world case (monthly interest rate on a loan)
"""
from scipy.optimize import root_scalar


def make_npv(cash_flows):
    def npv(rate):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))

    def d_npv(rate):
        return sum(-t * cf / (1 + rate) ** (t + 1)
                   for t, cf in enumerate(cash_flows) if t > 0)
    return npv, d_npv


def solve_irr(cash_flows, x0=0.10, x1=0.05, x2=0.20):
    npv, d_npv = make_npv(cash_flows)
    newton = root_scalar(npv, fprime=d_npv, x0=x0, method="newton", xtol=1e-12)
    secant = root_scalar(npv, x0=x1, x1=x2, method="secant", xtol=1e-12)
    return newton, secant, npv


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
cash_flows = [-10000, 3000, 3500, 4000, 2500]
newton, secant, npv = solve_irr(cash_flows)

print(f"IRR using Newton-Raphson: {newton.root:.6%}")
print(f"IRR using Secant method: {secant.root:.6%}")
print(f"NPV at Newton solution: {npv(newton.root):.6f}")
print(f"Iterations -> Newton: {newton.iterations}, Secant: {secant.iterations}")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE
# (a) bigger final inflow (5000)   (b) different starting guesses
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
cf2 = [-10000, 3000, 3500, 4000, 5000]
n2, s2, _ = solve_irr(cf2)
print(f"Final inflow 5000: Newton IRR = {n2.root:.4%}, Secant IRR = {s2.root:.4%}")
for x0 in (0.01, 0.10, 0.50):
    n3, _, _ = solve_irr(cash_flows, x0=x0)
    print(f"Newton start x0={x0}: IRR={n3.root:.6%}, iterations={n3.iterations}")

# ---------------------------------------------------------------
# PART 3: NEW CASE - effective monthly interest rate on a loan
# Assumptions: a customer receives 50,000 today and repays 4,500 at the
# end of each of 12 months (total repaid 54,000). Find the monthly rate r
# where NPV = 0, then convert to an annual effective rate (1+r)^12 - 1.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - loan interest rate ===")
loan_flows = [-50000] + [4500] * 12
nl, sl, npv_l = solve_irr(loan_flows, x0=0.02, x1=0.01, x2=0.03)
r = nl.root
print(f"Monthly rate (Newton): {r:.6%}  | Secant: {sl.root:.6%}")
print(f"Nominal annual rate (12 x monthly): {12 * r:.4%}")
print(f"Effective annual rate: {(1 + r) ** 12 - 1:.4%}")
print(f"NPV at solution: {npv_l(r):.2e}")
print(f"Total interest paid: {4500 * 12 - 50000}")
