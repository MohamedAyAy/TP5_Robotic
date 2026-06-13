#!/usr/bin/env python3
"""
Script de calcul de la cinématique directe du BCR Arm.
Exercice 6 - TP Cinématique Directe 7-DOF
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ForwardKinematics(Node):
    def __init__(self):
        super().__init__("forward_kinematics")

        # Paramètres géométriques du robot
        self.L1        =  0.200
        self.L2_offset =  0.065
        self.L3        =  0.410
        self.L4_offset = -0.065
        self.L5        =  0.310
        self.L6_offset =  0.060
        self.L7        =  0.105

        # Subscriber pour les états des joints
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info("Noeud forward_kinematics démarré, en attente des joint_states...")

    # ------------------------------------------------------------------
    # Cinématique directe (déléguée à BCRKinematics)
    # ------------------------------------------------------------------
    def compute_forward_kinematics(self, q):
        """
        Calcule la cinématique directe complète du BCR Arm.

        Args:
            q : liste des 7 angles articulaires [q1, q2, ..., q7] (rad)

        Returns:
            T         : matrice de transformation 4x4 de l'effecteur (T_0_7)
            positions : liste des positions (x,y,z) de chaque joint
        """
        import sys, os
        _scripts = os.path.dirname(os.path.abspath(__file__))
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        from bcr_kinematics import BCRKinematics
        kin = BCRKinematics()

        # Recompute step-by-step to capture intermediate positions
        q = list(q)
        dh_params = [
            (q[0],  0.025,              0,                  np.pi / 2),
            (q[1],  kin.L1,             kin.L2_offset,     -np.pi / 2),
            (q[2],  0,                  0,                  np.pi / 2),
            (q[3],  kin.L3,             kin.L4_offset,     -np.pi / 2),
            (q[4],  0,                  0,                  np.pi / 2),
            (q[5],  kin.L5,             kin.L6_offset,     -np.pi / 2),
            (q[6],  kin.L7,             0,                  0),
        ]
        T = np.eye(4)
        positions = [np.array([0.0, 0.0, 0.0])]
        for params in dh_params:
            T = T @ kin.dh_matrix(*params)
            positions.append(T[:3, 3].copy())
        return T, positions

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg):
        """Callback déclenché à chaque réception d'un message JointState."""

        if len(msg.position) < 7:
            self.get_logger().warn(
                f"Reçu seulement {len(msg.position)} joints, 7 attendus."
            )
            return

        q = list(msg.position[:7])

        # Calcul de la cinématique directe
        T, positions = self.compute_forward_kinematics(q)

        # Position et orientation de l'effecteur
        pos = T[:3, 3]
        rot = T[:3, :3]

        # Angles d'Euler (roll, pitch, yaw) depuis la matrice de rotation
        roll  = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1]**2 + rot[2, 2]**2))
        yaw   = np.arctan2(rot[1, 0], rot[0, 0])

        self.get_logger().info(
            f"\n--- Cinématique Directe ---\n"
            f"  Angles (rad) : {[f'{qi:.3f}' for qi in q]}\n"
            f"  Position effecteur :\n"
            f"    x = {pos[0]:.4f} m\n"
            f"    y = {pos[1]:.4f} m\n"
            f"    z = {pos[2]:.4f} m\n"
            f"  Orientation (roll, pitch, yaw) :\n"
            f"    roll  = {np.degrees(roll):.2f} deg\n"
            f"    pitch = {np.degrees(pitch):.2f} deg\n"
            f"    yaw   = {np.degrees(yaw):.2f} deg\n"
            f"  Positions intermédiaires :\n"
            + "\n".join([
                f"    Joint {i} : ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                for i, p in enumerate(positions)
            ])
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()