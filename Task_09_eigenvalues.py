"""Task 09: Eigenvalues and Eigenvectors
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (eigenvector centrality of a 4-node network)
Part 2: Parameter change (add a link, add a new node)
Part 3: New real-world case (PageRank of a small web graph)
"""
import numpy as np

nodes = ["A", "B", "C", "D"]


def centrality(adj):
    """Return dominant eigenvalue and normalized eigenvector centrality."""
    vals, vecs = np.linalg.eig(adj)
    idx = np.argmax(np.real(vals))
    vec = np.abs(np.real(vecs[:, idx]))
    return np.real(vals[idx]), vec / vec.sum(), np.real(vecs[:, idx])


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
A = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.]
])
eigenvalues, eigenvectors = np.linalg.eig(A)
dominant_value, scores, v = centrality(A)

print("All eigenvalues:", np.round(np.real(eigenvalues), 4))
print("Dominant eigenvalue:", dominant_value)
print("Eigenvector-based importance:")
for node, score in zip(nodes, scores):
    print(f"{node}: {score:.4f}")
print("Verification error:", np.linalg.norm(A @ v - dominant_value * v))

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - add a direct link A-D (network becomes
# fully connected, K4), then add a fifth node E attached only to A.
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change ===")
A2 = A.copy()
A2[0, 3] = A2[3, 0] = 1.0
val2, sc2, _ = centrality(A2)
print("After adding link A-D: dominant eigenvalue =", round(val2, 4))
for n_, s in zip(nodes, sc2):
    print(f"  {n_}: {s:.4f}")

A3 = np.zeros((5, 5))
A3[:4, :4] = A
A3[0, 4] = A3[4, 0] = 1.0
val3, sc3, _ = centrality(A3)
print("After adding leaf node E linked to A: dominant eigenvalue =", round(val3, 4))
for n_, s in zip(nodes + ["E"], sc3):
    print(f"  {n_}: {s:.4f}")

# ---------------------------------------------------------------
# PART 3: NEW CASE - PageRank on a 4-page website
# Assumptions: directed links  P1->P2, P1->P3, P2->P3, P3->P1, P4->P3.
# Damping factor 0.85. PageRank = dominant eigenvector (eigenvalue 1) of
# the Google matrix G = d*M + (1-d)/n * J, where M is column-stochastic.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - PageRank ===")
pages = ["P1", "P2", "P3", "P4"]
links = {0: [1, 2], 1: [2], 2: [0], 3: [2]}  # from -> list of targets
n = 4
M = np.zeros((n, n))
for src, targets in links.items():
    for t in targets:
        M[t, src] = 1.0 / len(targets)

for d in (0.85, 0.50):
    G = d * M + (1 - d) / n * np.ones((n, n))
    vals, vecs = np.linalg.eig(G)
    i = np.argmax(np.real(vals))
    pr = np.abs(np.real(vecs[:, i]))
    pr = pr / pr.sum()
    print(f"Damping d={d}: dominant eigenvalue = {np.real(vals[i]):.6f}")
    for p, s in zip(pages, pr):
        print(f"  {p}: {s:.4f}")
