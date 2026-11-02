"""Task 03: Mathematical Functions and Vectorization
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (network latency, loop vs vectorized)
Part 2: Parameter change (array size N)
Part 3: New real-world case (image brightness / gamma correction)
"""
import numpy as np
import timeit

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original - one million latency values ===")
rng = np.random.default_rng(42)
latency_ms = rng.uniform(5, 250, 1_000_000)


def loop_version():
    squared = []
    for value in latency_ms:
        seconds = value / 1000.0
        squared.append(seconds ** 2)
    return squared


def vectorized_version():
    seconds = latency_ms / 1000.0
    return seconds ** 2


loop_time = timeit.timeit(loop_version, number=1)
vector_time = timeit.timeit(vectorized_version, number=3) / 3
result = vectorized_version()

print("First five squared latency values:", result[:5])
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time / vector_time:.1f}x")

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - vary the array size N
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change - array size ===")
for n in (1_000, 100_000, 2_000_000):
    data = rng.uniform(5, 250, n)

    def loop_n():
        out = []
        for v in data:
            out.append((v / 1000.0) ** 2)
        return out

    def vec_n():
        return (data / 1000.0) ** 2

    t_loop = timeit.timeit(loop_n, number=1)
    t_vec = timeit.timeit(vec_n, number=5) / 5
    print(f"N={n:>9,}: loop={t_loop:.5f}s vectorized={t_vec:.6f}s "
          f"speed-up={t_loop / t_vec:.0f}x")

# ---------------------------------------------------------------
# PART 3: NEW CASE - image brightness normalization + gamma correction
# Assumptions: a 1000 x 1000 grayscale image, 8-bit pixels (0-255).
# Gamma correction: out = 255 * (pixel/255) ** gamma, gamma = 0.5
# (brightens dark regions). Pixels are simulated with random numbers.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - image gamma correction ===")
image = rng.integers(0, 120, size=(1000, 1000)).astype(float)  # dark image
gamma = 0.5

t0 = timeit.default_timer()
corrected = 255.0 * (image / 255.0) ** gamma
elapsed = timeit.default_timer() - t0

print(f"Pixels processed: {image.size:,} in {elapsed:.4f} s (no Python loop)")
print(f"Original mean brightness : {image.mean():.2f}")
print(f"Corrected mean brightness: {corrected.mean():.2f}")
print(f"Corrected range: {corrected.min():.1f} - {corrected.max():.1f}")

# Identity check from the concept section: sin^2 + cos^2 = 1 everywhere
x = rng.uniform(0, 2 * np.pi, 1_000_000)
y = np.sin(x) ** 2 + np.cos(x) ** 2
print(f"max |sin^2+cos^2 - 1| over 1M values: {np.max(np.abs(y - 1)):.2e}")
