#!/usr/bin/env python3
"""
Script de test des mouvements du BCR Arm.
Exercice 5 - TP Cinématique Directe 7-DOF
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time


class ArmTester(Node):

    def __init__(self):
        super().__init__("arm_tester")

        self.publisher = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10,
        )

        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7",
        ]

        # Laisser le temps au contrôleur de démarrer
        self.get_logger().info("En attente du contrôleur (3s)...")
        time.sleep(3)

    # ------------------------------------------------------------------
    def send_trajectory(self, positions, duration_sec=3.0):
        """Envoyer une commande de trajectoire au contrôleur."""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9),
        )
        msg.points = [point]

        self.publisher.publish(msg)
        self.get_logger().info(
            f"Trajectoire envoyée : {[f'{p:.2f}' for p in positions]}"
        )

    # ------------------------------------------------------------------
    def test_sequence(self):
        """Exécuter une séquence complète de mouvements de test."""
        self.get_logger().info("=== Début de la séquence de test ===")

        # 1. Position home (tous les joints à 0)
        self.get_logger().info(">>> Position HOME")
        self.send_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        time.sleep(4)

        # 2. Test joint 1 seul (rotation de la base)
        self.get_logger().info(">>> Test Joint 1 (rotation base)")
        self.send_trajectory([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        time.sleep(4)

        # 3. Retour home
        self.send_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0)
        time.sleep(3)

        # 4. Position étendue
        self.get_logger().info(">>> Position étendue")
        self.send_trajectory([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0], 3.0)
        time.sleep(4)

        # 5. Position test complète
        self.get_logger().info(">>> Position test complète")
        self.send_trajectory([0.5, -0.5, 0.3, -0.7, 0.2, 0.4, 0.1], 3.0)
        time.sleep(4)

        # 6. Retour home final
        self.get_logger().info(">>> Retour HOME final")
        self.send_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        time.sleep(4)

        self.get_logger().info("=== Séquence terminée ! ===")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    tester = ArmTester()
    try:
        tester.test_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
