"""Task 05: Matrix Algebra and Matrix Operations
BIT 2118 Scientific Computing - Group Work

Part 1: Original program (rotate + scale a square)
Part 2: Parameter change (90 degree rotation, different scale)
Part 3: New real-world case (robot gripper path, homogeneous coordinates)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# PART 1: ORIGINAL PROGRAM
# ---------------------------------------------------------------
print("=== PART 1: Original ===")
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

angle = np.deg2rad(45)
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
scale = np.array([
    [2.0, 0.0],
    [0.0, 1.5]
])

transformation = rotation @ scale
transformed_points = points @ transformation.T

print("Original points:\n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points:\n", transformed_points)
print("Determinant:", np.linalg.det(transformation))

# ---------------------------------------------------------------
# PART 2: PARAMETER CHANGE - 90 degrees, scale (1, 3)
# ---------------------------------------------------------------
print("\n=== PART 2: Parameter change (90 deg, scale 1 x 3) ===")
angle2 = np.deg2rad(90)
rot2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                 [np.sin(angle2), np.cos(angle2)]])
scale2 = np.diag([1.0, 3.0])
T2 = rot2 @ scale2
out2 = points @ T2.T
print("Transformed points:\n", np.round(out2, 4))
print("Determinant:", round(np.linalg.det(T2), 4), "(area scale factor)")
# Order matters: scale then rotate vs rotate then scale
print("Same result if order swapped?", np.allclose(rot2 @ scale2, scale2 @ rot2))

# Matrix inverse check
inv = np.linalg.inv(transformation)
print("T @ T^-1 is identity:", np.allclose(transformation @ inv, np.eye(2)))

# ---------------------------------------------------------------
# PART 3: NEW CASE - robot gripper path
# Assumptions: a gripper follows 5 waypoints (x, y) in metres on a work
# table. The whole path must be rotated by 30 degrees about the origin and
# shifted by (2.0, 1.0) m. A 3x3 homogeneous matrix performs both at once.
# ---------------------------------------------------------------
print("\n=== PART 3: New case - robot gripper path ===")
path = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.4], [1.0, 0.4], [1.0, 0.0]])
theta = np.deg2rad(30)
tx, ty = 2.0, 1.0
H = np.array([
    [np.cos(theta), -np.sin(theta), tx],
    [np.sin(theta),  np.cos(theta), ty],
    [0.0, 0.0, 1.0]
])
path_h = np.hstack([path, np.ones((len(path), 1))])  # add homogeneous 1
new_path = (path_h @ H.T)[:, :2]
print("Homogeneous matrix:\n", np.round(H, 4))
print("Transformed waypoints:\n", np.round(new_path, 4))
d_old = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
d_new = np.linalg.norm(np.diff(new_path, axis=0), axis=1).sum()
print(f"Path length before: {d_old:.4f} m, after: {d_new:.4f} m (rigid motion keeps length)")
print("Determinant of rotation part:", round(np.linalg.det(H[:2, :2]), 4))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
closed = np.vstack([points, points[0]])
tclosed = np.vstack([transformed_points, transformed_points[0]])
ax[0].plot(closed[:, 0], closed[:, 1], "o-", label="Original square")
ax[0].plot(tclosed[:, 0], tclosed[:, 1], "s-", label="Rotated 45 + scaled")
ax[0].set_aspect("equal"); ax[0].grid(True); ax[0].legend(); ax[0].set_title("Part 1")
ax[1].plot(path[:, 0], path[:, 1], "o-", label="Original path")
ax[1].plot(new_path[:, 0], new_path[:, 1], "s-", label="Rotated 30 + shifted")
ax[1].set_aspect("equal"); ax[1].grid(True); ax[1].legend(); ax[1].set_title("Part 3: robot path")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "task_05_matrix.png"), dpi=150)
plt.show()
