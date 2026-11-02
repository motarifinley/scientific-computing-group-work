# Scientific Computing Group Work

**Course:** BIT 2118 Scientific Computing
**Tasks covered:** 2 to 20 (one Python file per task in `code/`)

Run any task from the repository root, for example `python code/Task_08_iterative_linear.py`. Install the libraries with `pip install -r requirements.txt`. Plots are saved to `figures/`.

Every task file has three parts: **Part 1** is the original program from the brief, **Part 2** changes at least one input or parameter, and **Part 3** solves a different but related real-world case. Numbers below come from actually running the code. Timing results (Tasks 3 and 7) vary from machine to machine.

---

## Task 2: NumPy Arrays and Numerical Computation
**File:** `code/Task_02_numpy_arrays.py`

**Inputs and operation.** 24 hourly temperatures in °C stored in one array. `mean`, `min` and `max` summarise the day, and `F = 1.8C + 32` converts the whole array at once.

**Output.** Mean 23.12 °C, minimum 17.50 °C, maximum 29.40 °C. The first six Fahrenheit values are 65.3, 64.4, 64.04, 63.5, 64.22 and 66.2. The day is coolest before dawn and warmest in the early afternoon.

**Parameter change (heat wave, +5 °C on every reading).** Mean rises to 28.12 °C, minimum to 22.50 °C and maximum to 34.40 °C. The range (11.90 °C) and standard deviation (4.174) do not change. Adding a constant moves the data but leaves its spread alone.

**New case: IoT soil-moisture sensor.**
- *Assumptions:* one reading per hour for 24 h, moisture in % volumetric water, irrigation needed below 30 %.
- *Input data:* 24 moisture readings from 38.0 % down to 27.2 % and back up to 38.8 %.
- *Output:* mean 33.76 %, minimum 27.2 % at hour 13, maximum 38.8 % at hour 23. Six hours (10 to 15) fall below 30 %, which is 25 % of the day.
- *Interpretation:* the soil dries during the day and recovers at night, so irrigation is needed only in the hot midday window. A wetness index scaled to 0 to 1 is computed in one vectorised expression.

---

## Task 3: Mathematical Functions and Vectorization
**File:** `code/Task_03_vectorization.py`

**Inputs and operation.** One million simulated latency values (5 to 250 ms). Each is divided by 1000 and squared, once with a Python loop and once with a single NumPy expression, timed with `timeit`.

**Output.** Loop 0.2266 s against vectorised 0.0034 s, about 67 times faster. Both give the same squared values (first one 0.03788 s²). Vectorisation moves the repeated work into compiled code.

**Parameter change (array size N).**

| N | Speed-up |
|---|---|
| 1,000 | 35x |
| 100,000 | 28x |
| 2,000,000 | 146x |

Vectorisation wins at every size, and the advantage tends to grow as the data gets larger.

**New case: image gamma correction.**
- *Assumptions:* a 1000 × 1000 grayscale 8-bit image (random dark pixels 0 to 119), gamma = 0.5, `out = 255·(p/255)^gamma`.
- *Output:* 1,000,000 pixels processed in 0.0024 s with no loop. Mean brightness rises from 59.45 to 115.80 and the range becomes 0 to 174.2.
- *Interpretation:* gamma below 1 brightens dark images. An identity check also confirms `sin² + cos² = 1` to within 2.2e-16 over one million values.

---

## Task 4: Floating-Point Arithmetic and Numerical Errors
**File:** `code/Task_04_numerical_errors.py`

**Inputs and operation.** `0.1 + 0.2` compared with `0.3`, √2 compared with 1.414, and the expression `(√(1+x) − 1)/x` in an unstable and a stable form.

**Output.** `0.1 + 0.2 = 0.30000000000000004`, so `==` gives False while `np.isclose` gives True. For 1.414 the absolute error is 0.00021356 and the relative error 0.0151 %. At x = 1e-12 the direct formula gives 0.50004445 while the stable one gives exactly 0.5, so the direct form loses accuracy through cancellation.

**Parameter change.**
- A better approximation of √2 (1.41421, then 1.4142135) reduces the relative error from 1.5e-4 to 2.5e-6, then 4.4e-8.
- At x = 1e-14 the direct formula is wrong (0.4885) and at x = 1e-16 it collapses to 0, while the stable form stays at 0.5.
- Adding 0.1 ten thousand times gives 999.9029 in float32 and 1000.0000000002 in float64.

**New case: banking, summing one million payments of 0.01.**
- *Assumptions:* each payment is exactly one cent, so the true total is 10,000.00.
- *Output:* naive float sum 10000.0000001719 (error 1.7e-7), `math.fsum` 10000.0 and `Decimal` exactly 10000.00.
- *Interpretation:* small round-off errors accumulate. Money calculations should use `Decimal` or `fsum`, and comparisons should use a tolerance.

---

## Task 5: Matrix Algebra and Matrix Operations
**File:** `code/Task_05_matrix_algebra.py` | **Figure:** `figures/task_05_matrix.png`

**Inputs and operation.** The four corners of a unit square, a 45° rotation matrix and a scaling matrix (2, 1.5). The two matrices are multiplied into one transformation, which is applied to all points.

**Output.** The corner (1,1) moves to (0.354, 2.475). The determinant is 3.0, equal to 2 × 1.5, so areas become three times larger. The plot shows the original square next to the rotated, stretched shape.

**Parameter change (90° rotation, scale 1 × 3).** Corners map to (0,0), (0,1), (−3,1), (−3,0) and the determinant is still 3. Swapping the order of rotation and scaling gives a different matrix, so order matters. The inverse check `T @ T⁻¹ = I` holds.

**New case: robot gripper path.**
- *Assumptions:* 5 waypoints in metres, rotated 30° about the origin and shifted by (2.0, 1.0) m using one 3×3 homogeneous matrix.
- *Output:* the path starts at (2.0, 1.0) and ends at (2.866, 1.5). Path length is 1.8000 m both before and after, and the rotation part has determinant 1.
- *Interpretation:* a rotation plus a shift is a rigid motion, so the shape and length are preserved.

---

## Task 6: Solution of Systems of Linear Equations
**File:** `code/Task_06_linear_systems.py`

**Inputs and operation.** The system `4x + 2y = 20` (CPU) and `8x + 12y = 72` (memory), solved as `Ax = b` with `np.linalg.solve`.

**Output.** 3 Type A servers and 4 Type B servers. The check `A @ x` reproduces (20, 72) and the residual is 0.

**Parameter change.**
- New totals of 32 CPU and 112 GB give 5 and 6 servers.
- Making the second row proportional to the first (`[8, 4]`) gives determinant 0 and NumPy raises "Singular matrix". No unique solution exists because the two equations carry the same information.

**New case: three-loop DC circuit (mesh currents).**
- *Assumptions:* R1 to R5 = 10, 20, 30, 40, 50 Ω, V1 = 12 V in loop 1, V2 = 5 V opposing in loop 3.
- *Output:* I1 = 465.409 mA, I2 = 98.113 mA, I3 = −11.950 mA. Residual about 1e-15 and condition number 6.22. The current through R2 is 367.296 mA.
- *Interpretation:* I3 is negative, so loop 3 current flows opposite to the assumed direction because of the opposing source. The low condition number means the solution is reliable.

---

## Task 7: Gaussian Elimination and LU Decomposition
**File:** `code/Task_07_gaussian_lu.py`

**Inputs and operation.** A 3×3 matrix factored once with `lu_factor`, then reused for three different right-hand sides with `lu_solve`.

**Output.** Solutions (5, 5, 5), (5.317, 5.268, 4.756) and (4.902, 5.610, 5.537). Residuals are near 1e-15 and the first agrees with `np.linalg.solve`.

**Parameter change (500×500 matrix, 200 right-hand sides).** 200 full solves took 0.6929 s while one LU factorisation plus 200 cheap solves took 0.0258 s, about 27 times faster, with identical answers.

**New case: spring-chain structure under different load cases.**
- *Assumptions:* fixed stiffness matrix K (N/mm) for four joints, three load cases in newtons.
- *Output:* displacements are (0.5, 1.0, 1.5, 2.0) mm for the light case, (1.3, 2.5, 3.6, 4.6) mm for the medium case and (2, 2, 2, 2) mm for the uneven case. Determinant of K is 1e8.
- *Interpretation:* a 200 N load on joint 1 moves the whole chain 2 mm as a block. Factoring K once is what makes many load cases cheap, as in engineering and finite-element work.

---

## Task 8: Iterative Methods for Linear Systems
**File:** `code/Task_08_iterative_linear.py` | **Figure:** `figures/task_08_convergence.png`

**Inputs and operation.** A tridiagonal system for five grid temperatures with boundaries 100 °C and 20 °C. Jacobi and Gauss-Seidel are run until the largest change is below 1e-8.

**Output.** Temperatures 86.67, 73.33, 60.00, 46.67 and 33.33 °C, matching the direct solution. Jacobi needs 150 iterations and Gauss-Seidel 75. On the semilog plot both curves fall as straight lines, which means steady geometric convergence, and Gauss-Seidel is about twice as fast.

**Parameter change.** Tolerance 1e-3 needs 70 and 35 iterations, 1e-12 needs 214 and 107, so each extra digit costs more sweeps. Boundaries of 200 °C and 50 °C give 175, 150, 125, 100 and 75 °C after 78 iterations.

**New case: heated metal rod with 20 interior points.**
- *Assumptions:* steady state, no internal heat source, ends held at 80 °C and 20 °C.
- *Output:* Jacobi 1243 iterations and Gauss-Seidel 630. Temperatures fall from 77.14 to 22.86 °C. Maximum error against the exact straight-line profile is 4.3e-5.
- *Interpretation:* temperature falls linearly along the rod. A finer grid needs many more iterations, which is why faster iterative methods matter for large models.

---

## Task 9: Eigenvalues and Eigenvectors
**File:** `code/Task_09_eigenvalues.py`

**Inputs and operation.** The adjacency matrix of a 4-node network. `np.linalg.eig` finds the dominant eigenvalue and its eigenvector, normalised into importance scores.

**Output.** Dominant eigenvalue 2.5616. Scores A 0.2192, B 0.2808, C 0.2808, D 0.2192. B and C connect to three nodes each, so they are the most influential. The check `Av = λv` has error 1.5e-15.

**Parameter change.**
- Adding a link A–D makes the network fully connected, the eigenvalue becomes 3.0 and all scores equal 0.25.
- Attaching a new node E only to A gives eigenvalue 2.6412. E scores lowest (0.0842) and A rises to 0.2223 while D falls to 0.1905.

**New case: PageRank on a 4-page website.**
- *Assumptions:* links P1→P2, P1→P3, P2→P3, P3→P1, P4→P3, damping factor 0.85 (then 0.5).
- *Output:* PageRank P3 0.3941, P1 0.3725, P2 0.1958, P4 0.0375. With damping 0.5 the scores become 0.3654, 0.3077, 0.2019 and 0.1250. The dominant eigenvalue is 1.
- *Interpretation:* P3 ranks highest because every other page links to it. P4 ranks lowest because nothing links to it. Lower damping spreads rank more evenly.

---

## Task 10: Root Finding, Bisection and Fixed-Point
**File:** `code/Task_10_bisection_fixed_point.py` | **Figure:** `figures/task_10_roots.png`

**Inputs and operation.** The heat-balance function `f(T) = T − 25 − 10e^(−T/20)` on the bracket [20, 40], solved by bisection and by fixed-point iteration `T = 25 + 10e^(−T/20)`.

**Output.** Both methods give 27.525211 °C. Bisection used 38 iterations and fixed-point 13. In the plot the curve crosses zero at the dashed line near 27.5 °C.

**Parameter change.** Tolerance 1e-3 needs 15 iterations, 1e-6 needs 25 and 1e-10 needs 38. A narrower bracket [24, 30] needs 36. Starting the fixed-point from 100 °C still converges to 27.5252 in 14 iterations. The bracket [30, 40] has no sign change, so bisection refuses to run.

**New case: business break-even.**
- *Assumptions:* price 50, variable cost 20 per unit, fixed cost 2000, extra congestion cost 0.01q².
- *Output:* break-even at 68.22 units and 2931.78 units. The most profitable volume is 1500 units, with profit 20,500.
- *Interpretation:* the business loses money below about 68 units and above about 2932 units, and profit peaks between the two.

---

## Task 11: Root Finding, Newton-Raphson and Secant
**File:** `code/Task_11_newton_secant.py`

**Inputs and operation.** Cash flows (−10000, 3000, 3500, 4000, 2500). The internal rate of return is the rate where NPV = 0, found with Newton (using the derivative) and Secant.

**Output.** IRR 11.542461 % from both methods. Newton needed 4 iterations and Secant 7. NPV at the solution is 0.

**Parameter change.** Raising the last inflow to 5000 gives an IRR of 18.3273 %, since higher returns need a higher discount rate to cancel. Newton starting guesses of 0.01, 0.10 and 0.50 all find the same IRR, in 5, 4 and 8 iterations.

**New case: effective interest rate on a loan.**
- *Assumptions:* 50,000 received today and 4,500 repaid at the end of each of 12 months.
- *Output:* monthly rate 1.204346 %, nominal annual 14.4521 %, effective annual 15.4489 %. Total interest is 4,000.
- *Interpretation:* the effective annual rate is higher than 12 × monthly because of compounding, so the real cost of borrowing is above the nominal figure.

---

## Task 12: Polynomial and Nonlinear Equation Solving
**File:** `code/Task_12_nonlinear_equations.py`

**Inputs and operation.** The circle `x² + y² = 4` and the parabola `y = x² − 1` as two residual equations for `scipy.optimize.root`, with initial guesses (1, 1) and (−1, 1).

**Output.** Intersections at (1.5175, 1.3028) and (−1.5175, 1.3028), residuals near 1e-13, both converged.

**Parameter change.** Radius 1 gives intersections at (±1, 0). A poor start (0, −5) for the radius-2 case returns `converged = False` with residual 1.462, so the failure is detected. The polynomial `x² − 5x + 6` has roots 3 and 2.

**New case: projectile from a 2 m platform.**
- *Assumptions:* v₀ = 20 m/s upward, g = 9.81 m/s², no air drag, `h(t) = −4.905t² + 20t + 2`.
- *Output:* roots −0.0977 and 4.1751. The ball lands after 4.175 s, passes 15 m at 0.8115 s and 3.266 s, and peaks at 22.387 m at 2.039 s.
- *Interpretation:* the negative root is not physical. A quadratic can cross a given height twice, once on the way up and once on the way down.

---

## Task 13: Interpolation
**File:** `code/Task_13_interpolation.py` | **Figure:** `figures/task_13_interpolation.png`

**Inputs and operation.** Hourly temperatures from 08:00 to 14:00, estimated every 10 minutes with linear interpolation and a cubic spline.

**Output.** At 10:30 linear gives 23.25 °C and cubic gives 23.36 °C. In the plot the linear estimate is a set of straight segments, while the spline is a smooth curve through every measured point.

**Parameter change.** At 13:30 (near the peak) linear gives 25.25 °C and cubic 25.41 °C. When every second reading is removed, cubic errors at the missing hours (0.47, 0.03, 0.16) are generally smaller than linear (0.25, 0.25, 0.50). Asking for 15:00 is refused because it lies outside the data range, and interpolation does not extrapolate.

**New case: river-level gaps.**
- *Assumptions:* hourly gauge readings for 13 hours with hours 3, 4 and 9 missing, flood warning at 2.6 m.
- *Output:* estimates (cubic) 2.100 m, 2.276 m and 2.771 m. Water is at or above 2.6 m from hour 6 to hour 10, with an estimated peak of 2.785 m at hour 8.38.
- *Interpretation:* the gaps can be filled with reasonable values and the peak falls between samples, so the real maximum is slightly higher than the highest recorded reading.

---

## Task 14: Curve Fitting and Approximation
**File:** `code/Task_14_curve_fitting.py` | **Figure:** `figures/task_14_curve_fit.png`

**Inputs and operation.** Server response time at 50 to 350 users, fitted with the quadratic `ax² + bx + c` by least squares (`curve_fit`).

**Output.** `y = 0.002167x² − 0.1345x + 85.00`, RMSE 1.55 ms, prediction at 400 users 377.86 ms. The fitted curve passes close to all points and bends upward, so delay grows faster as load grows.

**Parameter change (model degree).** Degree 1 gives RMSE 18.83 ms and predicts 312.9 ms, degree 2 gives 1.55 ms and 377.9 ms, degree 3 gives 0.14 ms and 387.9 ms. A straight line clearly under-fits. The cubic fits the existing points best but extrapolates differently, so a lower RMSE alone does not make a model trustworthy.

**New case: logistic growth of mobile-money subscribers.**
- *Assumptions:* 10 yearly counts (millions) from a logistic curve plus small noise, model `K / (1 + e^(−r(t−t0)))`.
- *Output:* capacity K = 39.45 M, growth rate r = 0.851, midpoint t0 = 4.99, RMSE 0.43 M, forecast for year 12 is 39.35 M.
- *Interpretation:* growth is fastest around year 5 and then levels off near 39 to 40 million, so the market saturates.

---

## Task 15: Numerical Differentiation
**File:** `code/Task_15_numerical_differentiation.py` | **Figure:** `figures/task_15_differentiation.png`

**Inputs and operation.** GPS positions at 0 to 6 s. Speed is estimated with a forward difference at the start, central differences inside and a backward difference at the end.

**Output.** Speed rises from 4.80 m/s at t = 0 to 8.00 m/s at t = 6. The graph slopes upward, so the vehicle is speeding up.

**Parameter change (step size h on sin x at x = 1).** The central difference error falls from 8.6e-2 at h = 1 to 2.8e-11 at h = 1e-6, much faster than the forward difference (4.2e-7 at the same step). Below that, round-off dominates: both errors rise to 3.7e-3 at h = 1e-14. There is a best step size, not simply the smallest one.

**New case: train acceleration from velocity.**
- *Assumptions:* speed logged every 5 s as a train leaves, cruises and brakes, comfort limit |a| ≤ 1.0 m/s².
- *Output:* maximum acceleration 1.0 m/s² and maximum deceleration −1.4 m/s². Braking at 50, 55 and 60 s exceeds the comfort limit.
- *Interpretation:* acceleration is gentle, but the braking phase is harsher than passengers would find comfortable and would need to be spread over a longer time.

---

## Task 16: Numerical Integration
**File:** `code/Task_16_numerical_integration.py`

**Inputs and operation.** Smart-meter power (kW) every two hours over 12 h, integrated with the trapezoidal rule and Simpson's rule.

**Output.** Trapezoid 15.700 kWh, Simpson 15.800 kWh, difference 0.100 kWh. Energy is the area under the power curve.

**Parameter change.** With only 4-hourly data the estimates drift apart (15.400 and 15.967 kWh), so fewer readings mean more uncertainty. On the known integral of sin x over 0 to π (exact 2), 33 points give a trapezoid error of 1.6e-3 and a Simpson error of 1.0e-6. Adaptive `quad` returns 2.000000000000.

**New case: water volume from a flow meter.**
- *Assumptions:* flow rate (m³/s) every 10 min for 2 h (13 readings), minutes converted to seconds.
- *Output:* volume 25,440 m³ by both methods, average flow 3.533 m³/s.
- *Interpretation:* the two rules agree exactly here because the flow curve is smooth and symmetric. Agreement between two methods is a useful sanity check.

---

## Task 17: ODEs, Euler Method
**File:** `code/Task_17_euler_ode.py` | **Figure:** `figures/task_17_euler.png`

**Inputs and operation.** Newton's law of cooling `dT/dt = −k(T − T_ambient)` with T₀ = 90 °C, ambient 25 °C, k = 0.12 and step h = 1 min, stepped with Euler's method and compared with the exact solution.

**Output.** After 30 min Euler gives 26.40 °C and the exact value is 26.78 °C, an error of 0.372 °C. The temperature falls quickly at first and then slowly approaches 25 °C. The Euler curve sits slightly below the exact one.

**Parameter change (step size).** Errors: h = 5 gives 1.51 °C, h = 1 gives 0.372 °C, h = 0.1 gives 0.0383 °C, h = 0.01 gives 0.00384 °C. Reducing h by 10 reduces the error about 10 times. With h = 20 (above 2/k ≈ 16.7 min) the method is unstable: values go 90, −66, 152.4, −153.4 and oscillate.

**New case: logistic fish population.**
- *Assumptions:* P(0) = 500, r = 0.5 per year, carrying capacity 10,000, step 0.5 year, 20 years.
- *Output:* Euler 9995.3 against exact 9991.4 after 20 years. Maximum error during the run is 512 fish, and the population passes 5000 at about 6.5 years.
- *Interpretation:* growth is S-shaped and levels off at the carrying capacity. The Euler error is largest during fast growth.

---

## Task 18: Numerical Solution of ODEs Using SciPy
**File:** `code/Task_18_scipy_ode.py` | **Figure:** `figures/task_18_ode.png`

**Inputs and operation.** RC charging circuit `dV/dt = (Vs − V)/(RC)` with Vs = 5 V, R = 1000 Ω, C = 0.001 F (time constant 1 s), solved by `solve_ivp`.

**Output.** 4.9663 V at 5 s, with a maximum error against the exact solution of 9.3e-10 V. The curve rises steeply and flattens towards 5 V.

**Parameter change.** Halving the capacitance to 0.0005 F gives V(1 s) = 4.3232 V instead of 3.1605 V, so charging is faster. At one time constant the voltage is 63.2 % of the supply. Discharging from 5 V gives 1.8395 V at 1 s and 0.0337 V at 5 s.

**New case: SIR epidemic in a school.**
- *Assumptions:* 1000 students, 1 initial case, contact rate 0.5 per day, recovery rate 0.2 per day (R0 = 2.5), 120 days.
- *Output:* peak of 234 infected on day 24.4 and 893 students infected in total. With distancing (contact rate 0.25, R0 = 1.25) the peak drops to 22 and the total to 294. S + I + R stays at 1000.
- *Interpretation:* halving contacts flattens the epidemic dramatically, which shows why modest behaviour changes matter.

---

## Task 19: Numerical Optimization
**File:** `code/Task_19_optimization.py`

**Inputs and operation.** Minimise cost `12x + 18y` subject to capacity `100x + 170y ≥ 1000` and `x, y ≥ 0` using SLSQP.

**Output.** Type A = 0 and Type B = 5.882 for a cost of 105.88. Type B is cheaper per unit of capacity (0.1059 against 0.12), so the optimiser uses only B. With whole servers, the best plan is 6 Type B servers at cost 108.

**Parameter change.** If B costs 25, A becomes cheaper per capacity and the optimum switches to 10 Type A servers at cost 120. If the capacity requirement rises to 1500, the continuous cost is 158.82 and the whole-server plan is 9 Type B servers at cost 162.

**New case: delivery fleet.**
- *Assumptions:* vans cost 40/day (30 parcels, 400 kg), lorries cost 90/day (80 parcels, 1200 kg), need at least 500 parcels and 8000 kg.
- *Output:* continuous optimum 6.667 lorries and no vans at cost 600. Whole vehicles: 2 vans and 6 lorries at cost 620.
- *Interpretation:* the weight constraint is the one that binds. Rounding to whole vehicles costs 20 more than the continuous optimum.

---

## Task 20: Random Numbers and Monte Carlo Methods
**File:** `code/Task_20_monte_carlo.py` | **Figure:** `figures/task_20_monte_carlo.png`

**Inputs and operation.** Server demand drawn from a normal distribution (mean 850, standard deviation 120) for 100,000 scenarios. The overload probability is the fraction of scenarios above the capacity of 1000.

**Output.** Estimated overload probability 10.6160 % (10,616 of 100,000 cases). The exact value for this distribution is 10.565 %. In the convergence plot the estimate swings widely at small N and settles near the exact value as N grows.

**Parameter change.** Raising mean demand to 900 gives 20.15 % (exact 20.23 %) and to 950 gives 33.90 % (exact 33.85 %). Using more simulations reduces the error: N = 100 gives 0.435 percentage points of error, N = 10,000 gives 0.135 and N = 1,000,000 gives 0.010.

**New cases.**
- *Estimating π:* N = 1,000 gives 3.24000 (error 0.098), N = 100,000 gives 3.13320 (0.0084) and N = 5,000,000 gives 3.14220 (0.0006).
- *Project-delay risk.* Assumptions: three sequential tasks with triangular durations in days (design 5/8/14, build 10/15/30, test 4/6/12), deadline 32 days, 200,000 simulations. Output: mean duration 34.68 days (standard deviation 4.96), 67.6 % chance of finishing after 32 days, 90th percentile 41.5 days, and 43.4 days needed for 95 % confidence. Interpretation: the 32-day deadline is more likely to be missed than met, and the long right tail of the histogram comes from the build task's upper bound.
