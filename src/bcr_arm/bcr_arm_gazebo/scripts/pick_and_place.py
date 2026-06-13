#!/usr/bin/env python3
"""
Full Pick-and-Place Pipeline (ROS 2 Node)
==========================================
Orchestrates: perception → segmentation → DGCNN classification → IK → trajectory.

Usage:
    ros2 run bcr_arm_gazebo pick_and_place.py
"""

import os
import sys
import time
import numpy as np

# ------------------------------------------------------------------
# Ensure the scripts directory is in PYTHONPATH so that relative
# imports (segmentation, classifier, ik_solver, trajectory) work.
# ------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
# Note: do NOT inject the UV venv (.venv/python3.10) into sys.path here.
# The UV venv's compiled .so extensions are built for Python 3.10 and will
# crash when loaded by the system Python 3.12 used by ros2 run.
# System Python 3.12 already has: torch, numpy, scipy, sklearn — all we need.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sensor_msgs_py.point_cloud2 as pc2

from segmentation import segment_object
from classifier  import ObjectClassifier
from ik_solver   import inverse_kinematics
from trajectory  import quintic_trajectory

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
HOME         = np.zeros(7)
PRE_GRASP_DZ = 0.05   # 5 cm above centroid (already added by segment_object)
T_MOVE       = 3.0    # trajectory duration [s]

# Weights path — set via env var or default workspace location
_WEIGHTS_PATH = os.environ.get(
    "DGCNN_WEIGHTS",
    os.path.normpath(os.path.join(_SCRIPTS_DIR, "../../../../models/dgcnn_modelnet40.pth"))
)


class PickAndPlace(Node):
    """ROS 2 node that runs the pick-and-place pipeline once."""

    def __init__(self):
        super().__init__("pick_and_place")

        self.joint_names = [f"joint{i}" for i in range(1, 8)]

        # Publisher: Joint Trajectory Controller
        self.cmd_pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10,
        )

        # Subscription: depth camera point cloud
        self.cloud = None
        self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_cb, 10
        )

        # DGCNN Classifier
        self.get_logger().info(f"Loading DGCNN weights from: {_WEIGHTS_PATH}")
        self.classifier = ObjectClassifier(weights_path=_WEIGHTS_PATH)

        # Timer: run pipeline once after 3 s
        self.create_timer(3.0, self.run_once)
        self.done = False

    # ------------------------------------------------------------------
    def cloud_cb(self, msg: PointCloud2):
        """Convert PointCloud2 to world-frame Nx3 array.

        Camera pose: xyz='0.6 0 1.2'  rpy='0 1.5708 0'  (pure pitch 90 deg)
        Rotation matrix R (camera→world) for pitch=pi/2:
          X_world =  Z_cam + 0.6
          Y_world =  Y_cam + 0.0
          Z_world = -X_cam + 1.2
        """
        raw = list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                   skip_nans=True))
        if not raw:
            return
        cam = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float64)
        # Transform to world frame
        world = np.column_stack([
            cam[:, 2] + 0.6,   # X_world
            cam[:, 1],          # Y_world
            -cam[:, 0] + 1.2,  # Z_world
        ]).astype(np.float32)
        # Crop point cloud to the region of interest (table top workspace)
        # Table top is at Z=0.425, cup is on it.
        # Box: X in [0.2, 0.8], Y in [-0.3, 0.3], Z in [0.41, 0.65]
        roi = (
            (world[:, 0] >= 0.2) & (world[:, 0] <= 0.8) &
            (world[:, 1] >= -0.3) & (world[:, 1] <= 0.3) &
            (world[:, 2] >= 0.41) & (world[:, 2] <= 0.65)
        )
        world = world[roi]
        
        # Keep only finite points
        finite = np.isfinite(world).all(axis=1)
        world = world[finite]
        if world.shape[0] > 10:
            self.cloud = world
            self.get_logger().info(
                f"Cloud received: {world.shape[0]} pts  "
                f"| Z range [{world[:,2].min():.3f}, {world[:,2].max():.3f}]",
                once=True
            )

    # ------------------------------------------------------------------
    def send_trajectory(self, q_from, q_to, T: float = T_MOVE):
        """Build and publish a quintic joint trajectory."""
        times, positions = quintic_trajectory(q_from, q_to, T=T)
        msg              = JointTrajectory()
        msg.joint_names  = self.joint_names

        for t, q in zip(times, positions):
            pt              = JointTrajectoryPoint()
            pt.positions    = q.tolist()
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9)
            )
            msg.points.append(pt)

        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f"Trajectory sent ({len(msg.points)} pts, duration={T:.1f} s)"
        )
        time.sleep(T + 0.5)

    # ------------------------------------------------------------------
    def run_once(self):
        """Main pipeline — called once after Gazebo stabilises."""
        if self.done or self.cloud is None:
            if self.cloud is None:
                self.get_logger().warn("Waiting for point cloud…")
            return
        self.done = True
        self.get_logger().info("=== PIPELINE START ===")

        # 1. Segmentation (RANSAC plane + DBSCAN cluster)
        obj_pts, centroid = segment_object(self.cloud)
        if obj_pts is None:
            self.get_logger().error("No object detected in point cloud.")
            return
        self.get_logger().info(f"Object centroid: {centroid.round(4)}")

        # 2. Classification with DGCNN
        name, label = self.classifier.predict(obj_pts)
        self.get_logger().info(f"Predicted class: {name}  (label={label})")

        # 3. IK target: centroid (segment_object already adds PRE_GRASP_DZ)
        x_target = centroid.copy()

        # 4. Inverse Kinematics
        q_target, ok, n_iter = inverse_kinematics(HOME, x_target)
        if not ok:
            self.get_logger().error(
                f"IK did not converge after {n_iter} iterations."
            )
            return
        self.get_logger().info(
            f"IK converged in {n_iter} iter — q={q_target.round(3)}"
        )

        # 5. Move HOME → pre-grasp pose
        self.send_trajectory(HOME, q_target)

        # 6. Return HOME
        self.send_trajectory(q_target, HOME)
        self.get_logger().info("=== PIPELINE DONE ===")


# ------------------------------------------------------------------
def main():
    rclpy.init()
    node = PickAndPlace()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
