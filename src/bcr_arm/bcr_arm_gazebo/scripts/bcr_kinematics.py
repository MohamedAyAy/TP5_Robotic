#!/usr/bin/env python3
"""
BCR Arm Kinematics (standalone, no ROS dependency)
====================================================
Pure-Python class for DH-based forward kinematics.
Used by ik_solver.py and any standalone test scripts.
"""

import numpy as np


class BCRKinematics:
    """
    7-DOF BCR Arm forward kinematics via Denavit–Hartenberg convention.

    DH table (modified DH):
      Joint | theta  | d              | a              | alpha
      ------|--------|----------------|----------------|--------
        J1  |  q[0]  | 0.025          | 0              | +π/2
        J2  |  q[1]  | L1 = 0.200     | L2_off = 0.065 | -π/2
        J3  |  q[2]  | 0              | 0              | +π/2
        J4  |  q[3]  | L3 = 0.410     | L4_off=-0.065  | -π/2
        J5  |  q[4]  | 0              | 0              | +π/2
        J6  |  q[5]  | L5 = 0.310     | L6_off= 0.060  | -π/2
        J7  |  q[6]  | L7 = 0.105     | 0              |  0
    """

    L1        =  0.200
    L2_offset =  0.065
    L3        =  0.410
    L4_offset = -0.065
    L5        =  0.310
    L6_offset =  0.060
    L7        =  0.105

    @staticmethod
    def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Standard DH transformation matrix (4×4)."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ],
        ])

    def compute_forward_kinematics(self, q) -> np.ndarray:
        """
        Compute the end-effector homogeneous transform T_0_7.

        Args:
            q: array-like of 7 joint angles (rad)

        Returns:
            T: (4, 4) homogeneous transformation matrix (world → EEF)
        """
        q = np.asarray(q, dtype=float)
        dh_params = [
            (q[0],  0.025,              0,                  np.pi / 2),
            (q[1],  self.L1,            self.L2_offset,    -np.pi / 2),
            (q[2],  0,                  0,                  np.pi / 2),
            (q[3],  self.L3,            self.L4_offset,    -np.pi / 2),
            (q[4],  0,                  0,                  np.pi / 2),
            (q[5],  self.L5,            self.L6_offset,    -np.pi / 2),
            (q[6],  self.L7,            0,                  0),
        ]
        T = np.eye(4)
        for params in dh_params:
            T = T @ self.dh_matrix(*params)
        return T


# Singleton for use by ik_solver
_kinematics = BCRKinematics()


def fk_position(q) -> np.ndarray:
    """Convenience wrapper: returns (x, y, z) of end-effector for joint config q."""
    T = _kinematics.compute_forward_kinematics(q)
    return T[:3, 3]
