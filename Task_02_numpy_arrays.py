"""Task 02: NumPy Arrays and Numerical Computation
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (weather-station temperatures)
Part 2: Parameter change (heat-wave shift of +5 C)
Part 3: New real-world case (IoT soil-moisture sensor)
"""
import numpy as np

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original - 24 hourly temperatures ===")
temperatures_c = np.array([
    18.5, 18.0, 17.8, 17.5, 17.9, 19.0,
    21.5, 23.8, 25.6, 27.1, 28.3, 29.0,
    29.4, 29.1, 28.5, 27.6, 26.2, 24.8,
    23.4, 22.1, 21.0, 20.2, 19.6, 19.0
])

mean_temp = np.mean(temperatures_c)
min_temp = np.min(temperatures_c)
max_temp = np.max(temperatures_c)

# Vectorized Celsius-to-Fahrenheit conversion
temperatures_f = 1.8 * temperatures_c + 32

print(f"Daily mean temperature: {mean_temp:.2f} C")
print(f"Minimum temperature: {min_temp:.2f} C")
print(f"Maximum temperature: {max_temp:.2f} C")
print("First six Fahrenheit readings:", temperatures_f[:6])

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - a heat wave adds 5 C to every reading
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - heat wave (+5 C) ===")
hot = temperatures_c + 5.0
print(f"Mean: {hot.mean():.2f} C (was {mean_temp:.2f} C)")
print(f"Min : {hot.min():.2f} C (was {min_temp:.2f} C)")
print(f"Max : {hot.max():.2f} C (was {max_temp:.2f} C)")
print(f"Range (max-min): {np.ptp(hot):.2f} C (was {np.ptp(temperatures_c):.2f} C)")
print(f"Standard deviation: {hot.std():.3f} (was {temperatures_c.std():.3f})")

# ---------------------------------------------------------------
# PART 3: NEW CASE - IoT soil-moisture sensor (percent volumetric water)
# Assumptions: 1 reading per hour for 24 h; irrigation is needed
# whenever moisture drops below 30 %; sensor range is 0-100 %.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - soil moisture IoT sensor ===")
moisture = np.array([
    38.0, 37.5, 37.0, 36.2, 35.5, 34.8,
    34.0, 33.0, 31.5, 30.2, 29.0, 28.1,
    27.5, 27.2, 27.8, 29.5, 32.0, 34.5,
    36.0, 37.2, 38.0, 38.4, 38.6, 38.8
])
threshold = 30.0
below = moisture < threshold

print(f"Mean moisture: {moisture.mean():.2f} %")
print(f"Min moisture : {moisture.min():.2f} % at hour {moisture.argmin()}")
print(f"Max moisture : {moisture.max():.2f} % at hour {moisture.argmax()}")
print(f"Hours below {threshold:.0f} %: {below.sum()} of {moisture.size}")
print("Hours needing irrigation:", np.where(below)[0])
print(f"Percentage of day below threshold: {100 * below.mean():.1f} %")
# Normalize to a 0-1 'wetness index' with one vectorized expression
wetness = (moisture - moisture.min()) / np.ptp(moisture)
print("Wetness index (first 6 h):", np.round(wetness[:6], 3))
