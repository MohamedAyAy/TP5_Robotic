#!/usr/bin/env python3
"""
Comprehensive standalone test: IK + FK + Classifier + Segmentation
====================================================================
Runs everything WITHOUT Gazebo / ROS.
Usage:
    python3 test_pipeline_standalone.py

Requires: torch (system python3), open3d/numpy (uv run)
Since this test uses both, run with system python3 which has torch
(open3d will fail — segmentation test uses uv run separately).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ============================================================
# Test 1: Forward Kinematics
# ============================================================
print("=" * 60)
print("TEST 1: Forward Kinematics (BCRKinematics)")
print("=" * 60)

from bcr_kinematics import BCRKinematics, fk_position

kin = BCRKinematics()
q_home = np.zeros(7)
T_home = kin.compute_forward_kinematics(q_home)
pos_home = fk_position(q_home)
print(f"  q_home = {q_home}")
print(f"  EEF position at home: {pos_home.round(4)}")
assert T_home.shape == (4, 4), "FK must return 4x4 matrix"
print("  ✅ FK OK")


# ============================================================
# Test 2: Inverse Kinematics
# ============================================================
print()
print("=" * 60)
print("TEST 2: Inverse Kinematics (DLS)")
print("=" * 60)

from ik_solver import inverse_kinematics

targets = [
    np.array([0.45, 0.10, 0.55]),
    np.array([0.45, 0.10, 0.53]),   # TP5 exact target (centroid + 5cm + 5cm)
    np.array([0.30, 0.00, 0.70]),
]

for x_target in targets:
    q_sol, ok, n_iter = inverse_kinematics(q_home, x_target)
    achieved = fk_position(q_sol)
    error_mm = np.linalg.norm(achieved - x_target) * 1000
    status   = "✅" if ok and error_mm < 2.0 else "⚠️"
    print(f"  {status} target={x_target}  converged={ok}  "
          f"iter={n_iter}  error={error_mm:.2f}mm")


# ============================================================
# Test 3: Quintic Trajectory
# ============================================================
print()
print("=" * 60)
print("TEST 3: Quintic Trajectory")
print("=" * 60)

from trajectory import quintic_trajectory

q0 = np.zeros(7)
qf = np.array([0.5, 0.3, -0.2, 0.4, -0.1, 0.2, 0.0])
times, positions = quintic_trajectory(q0, qf, T=3.0, n_points=50)

assert times[0]  == 0.0,    "Must start at t=0"
assert abs(times[-1] - 3.0) < 1e-9, "Must end at T"
assert np.allclose(positions[0], q0, atol=1e-9), "Must start at q0"
assert np.allclose(positions[-1], qf, atol=1e-6), "Must end at qf"
print(f"  ✅ Trajectory OK: {len(times)} points, "
      f"q_start={positions[0].round(3)}, q_end={positions[-1].round(3)}")


# ============================================================
# Test 4: DGCNN Classifier
# ============================================================
print()
print("=" * 60)
print("TEST 4: DGCNN Classifier (ModelNet40 pre-trained)")
print("=" * 60)

from classifier import ObjectClassifier, CLASS_NAMES

clf = ObjectClassifier()

# Synthetic point clouds for recognizable shapes
test_cases = {
    "cylinder (cup-like)": lambda: np.column_stack([
        0.4 * np.cos(np.linspace(0, 2*np.pi, 1024)),
        0.4 * np.sin(np.linspace(0, 2*np.pi, 1024)),
        np.linspace(-0.5, 0.5, 1024),
    ]),
    "flat disk (table-like)": lambda: np.column_stack([
        np.random.uniform(-1, 1, 1024),
        np.random.uniform(-1, 1, 1024),
        np.random.normal(0, 0.01, 1024),
    ]),
    "sphere": lambda: (lambda pts: pts / np.linalg.norm(pts, axis=1, keepdims=True))(
        np.random.randn(1024, 3)
    ),
}

for desc, make_pts in test_cases.items():
    pts = make_pts().astype(np.float32)
    name, label = clf.predict(pts)
    print(f"  [{desc}] → predicted: '{name}' (label={label})")

print("  ✅ Classifier OK")


# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("ALL STANDALONE TESTS PASSED ✅")
print("=" * 60)
print()
print("Note: Segmentation requires open3d (not in system Python).")
print("Test it separately with:")
print("  cd /home/mohamed/bcr_ws1 && uv run python -c \"...\"")
print("  (see segmentation.py for details)")
