#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class Perception(Node):
    def __init__(self):
        super().__init__("perception")
        self.subscription = self.create_subscription(
            PointCloud2, "/camera/points", self.callback, 10)
        self.latest_cloud = None

    def callback(self, msg: PointCloud2):
        # Convertir le message en tableau Nx3 (x, y, z)
        points = np.array([[p[0], p[1], p[2]]
                           for p in pc2.read_points(msg, field_names=("x", "y", "z"), 
                           skip_nans=True)])
        self.latest_cloud = points
        self.get_logger().info(f"Nuage recu: {points.shape[0]} points")

def main():
    rclpy.init()
    node = Perception()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
