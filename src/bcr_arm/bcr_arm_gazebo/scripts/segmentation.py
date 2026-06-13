#!/usr/bin/env python3
"""
Point Cloud Segmentation
========================
Removes the table plane (RANSAC) and isolates the main object cluster (DBSCAN).

Two backends are supported:
  - Open3D (preferred, higher performance) — available via `uv run python`
  - Pure NumPy + SciPy + scikit-learn (fallback, available system-wide via python3)

The function `segment_object` transparently selects the best available backend.

Standalone test:
    python3 segmentation.py              # uses numpy+scipy+sklearn backend
    uv run python segmentation.py        # uses open3d backend
"""

import numpy as np


# ───────────────────────────────────────────────────────────────────────────────
# Open3D backend (best quality)
# ───────────────────────────────────────────────────────────────────────────────

def _segment_open3d(points_np: np.ndarray, distance_threshold: float):
    """Segmentation using Open3D (RANSAC plane + DBSCAN cluster)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # 1. RANSAC: remove table plane
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=500,
    )
    object_pcd = pcd.select_by_index(inliers, invert=True)

    # 2. DBSCAN: find main cluster
    labels = np.array(object_pcd.cluster_dbscan(eps=0.02, min_points=20))
    if labels.max() < 0:
        return None, None

    main_label = np.bincount(labels[labels >= 0]).argmax()
    object_pts = np.asarray(object_pcd.points)[labels == main_label]

    centroid = object_pts.mean(axis=0)
    return object_pts, centroid


# ───────────────────────────────────────────────────────────────────────────────
# Pure-Python fallback backend (numpy + scipy + sklearn)
# ───────────────────────────────────────────────────────────────────────────────

def _ransac_plane(points: np.ndarray, threshold: float, n_iter: int = 500):
    """
    RANSAC plane fitting.

    Returns:
        inlier_mask: boolean array, True for inliers (table plane)
    """
    best_mask = np.zeros(len(points), dtype=bool)
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]

        # Plane normal via cross product
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -np.dot(n, p0)

        # Distance of all points to the plane
        dist = np.abs(points @ n + d)
        mask = dist < threshold

        if mask.sum() > best_mask.sum():
            best_mask = mask

    return best_mask


def _segment_numpy(points_np: np.ndarray, distance_threshold: float):
    """Segmentation using pure numpy + scipy + scikit-learn."""
    from sklearn.cluster import DBSCAN

    # 1. RANSAC: identify and remove table plane
    plane_mask = _ransac_plane(points_np, threshold=distance_threshold)
    object_pts = points_np[~plane_mask]

    if len(object_pts) < 20:
        return None, None

    # 2. DBSCAN: cluster non-table points
    db = DBSCAN(eps=0.02, min_samples=20, n_jobs=-1)
    labels = db.fit_predict(object_pts)

    valid = labels[labels >= 0]
    if len(valid) == 0:
        return None, None

    main_label = np.bincount(valid).argmax()
    cluster_pts = object_pts[labels == main_label]

    centroid = cluster_pts.mean(axis=0)
    return cluster_pts, centroid


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def segment_object(points_np: np.ndarray, distance_threshold: float = 0.01):
    """
    Segment the main object from a point cloud by removing the table plane
    (RANSAC) and finding the largest non-table cluster (DBSCAN).

    The centroid returned already includes a +5 cm pre-grasp offset in Z
    (matching the TP5 pick-and-place specification).

    Args:
        points_np:          (N, 3) array of XYZ points
        distance_threshold: RANSAC plane inlier distance [m]

    Returns:
        object_pts: (M, 3) points of the main object  (None if not found)
        centroid:   (3,) pre-grasp centroid            (None if not found)
    """
    # Sanitize the point cloud: remove NaN and Inf values
    valid_mask = np.isfinite(points_np).all(axis=1)
    points_np = points_np[valid_mask]

    if len(points_np) < 3:
        return None, None

    try:
        obj_pts, centroid = _segment_open3d(points_np, distance_threshold)
        backend = "open3d"
    except ImportError:
        obj_pts, centroid = _segment_numpy(points_np, distance_threshold)
        backend = "numpy+sklearn"

    if obj_pts is None:
        return None, None

    # Add vertical pre-grasp offset (5 cm above object centroid)
    PRE_GRASP_DZ = 0.05
    centroid = centroid.copy()
    centroid[2] += PRE_GRASP_DZ

    return obj_pts, centroid


# ───────────────────────────────────────────────────────────────────────────────
# Standalone test
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing segment_object with synthetic table + cup scene…")
    rng = np.random.default_rng(42)

    # Table: flat plane at z ≈ 0
    N_table = 4000
    table = np.column_stack([
        rng.uniform(-0.4, 0.9, N_table),
        rng.uniform(-0.4, 0.4, N_table),
        rng.normal(0.0, 0.002, N_table),
    ])

    # Cup: cylinder at (0.45, 0.10, 0.02–0.12)
    N_cup = 400
    theta = rng.uniform(0, 2 * np.pi, N_cup)
    cup = np.column_stack([
        0.45 + 0.04 * np.cos(theta) + rng.normal(0, 0.003, N_cup),
        0.10 + 0.04 * np.sin(theta) + rng.normal(0, 0.003, N_cup),
        rng.uniform(0.02, 0.12, N_cup),
    ])
    cloud = np.vstack([table, cup]).astype(np.float32)

    obj_pts, centroid = segment_object(cloud, distance_threshold=0.01)

    if obj_pts is not None:
        print(f"  Object points: {obj_pts.shape[0]}")
        print(f"  Centroid (with pre-grasp offset): {centroid.round(4)}")
        print("  ✅ Segmentation OK")
    else:
        print("  ❌ No object found")
