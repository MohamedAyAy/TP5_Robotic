#!/usr/bin/env python3
"""
Damped Least Squares (DLS) Inverse Kinematics Solver
=====================================================
Uses the standalone BCRKinematics module — NO ROS dependency.

Standalone test:
    python3 ik_solver.py
"""

import numpy as np
from bcr_kinematics import fk_position


def jacobian_fd(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Numerical Jacobian (3×7) by central finite differences.

    Args:
        q:   joint angles (7,)
        eps: perturbation step

    Returns:
        J: (3, 7) positional Jacobian
    """
    n = len(q)
    J = np.zeros((3, n))
    x0 = fk_position(q)
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        J[:, i] = (fk_position(q + dq) - x0) / eps
    return J


def inverse_kinematics(
    q_init,
    x_target,
    lambda_damp: float = 0.05,
    alpha: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 200,
):
    """
    DLS inverse kinematics for the 7-DOF BCR Arm.

    Args:
        q_init:      initial joint configuration (7,) [rad]
        x_target:    desired end-effector position (3,) [m]
        lambda_damp: DLS damping factor
        alpha:       step size (line-search factor)
        tol:         convergence tolerance [m]
        max_iter:    maximum iterations

    Returns:
        q:  final joint angles (7,)
        ok: True if converged
        it: number of iterations performed
    """
    q = np.array(q_init, dtype=float).copy()
    x_target = np.asarray(x_target, dtype=float)

    for it in range(max_iter):
        x_curr = fk_position(q)
        dx = x_target - x_curr

        if np.linalg.norm(dx) < tol:
            return q, True, it

        J = jacobian_fd(q)
        # DLS: dq = J^T (J J^T + λ² I)^{-1} dx
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(
            JJt + (lambda_damp ** 2) * np.eye(3), dx
        )
        q += alpha * dq

    return q, False, max_iter


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    q0 = np.zeros(7)
    x_target = np.array([0.45, 0.10, 0.55])

    print(f"Target position: {x_target}")
    print(f"Start FK:        {fk_position(q0)}")

    q_sol, ok, n = inverse_kinematics(q0, x_target)
    achieved = fk_position(q_sol)
    error = np.linalg.norm(achieved - x_target)

    print(f"\nConverged: {ok}  in {n} iterations")
    print(f"q_sol:     {q_sol.round(4)}")
    print(f"FK(q_sol): {achieved.round(4)}")
    print(f"Error:     {error*1000:.2f} mm")
