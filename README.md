# BCR 7-DOF Robotic Arm: 3D Perception & Pick-and-Place (TP5)

This repository contains the source code for the 3D Perception and Pick-and-Place pipeline of the BCR 7-DOF Robotic Arm, implemented under **ROS 2 Jazzy** and **Gazebo Harmonic (gz-sim8)**.

## Project Structure

* **`src/bcr_arm/bcr_arm_description/`**: Contains the URDF, meshes, and configuration files.
  * `urdf/sensors/depth_camera.urdf.xacro`: The updated depth camera configuration with fixed sensor plugins.
* **`src/bcr_arm/bcr_arm_gazebo/`**: Contains Gazebo launch scripts, world definitions, and control orchestrators.
  * `worlds/pick_scene.world`: The simulation scene featuring the table, target object, and the robot.
  * `launch/bcr_arm.gazebo.launch.py`: The modified launch script (runs Gazebo Sim server and GUI, spawns the robot from `/robot_description`).
  * `scripts/`: Python source scripts for execution:
    * `perception.py`: ROS 2 subscriber to point cloud data.
    * `segmentation.py`: RANSAC plane segmentation (to remove the table top) and DBSCAN clustering (to isolate the object).
    * `classifier.py`: Loads the ModelNet40 DGCNN classifier model.
    * `dgcnn_model.py`: PyTorch architecture definitions for DGCNN.
    * `ik_solver.py`: Damped Least Squares (DLS) Inverse Kinematics solver.
    * `trajectory.py`: Quintic joint trajectory generator.
    * `pick_and_place.py`: Main orchestrator node.
* **`setup_tp5.bash`**: Script to source ROS 2, configure workspace variables, and activate the `uv` virtual environment automatically.
* **`report_tp5.md`**: Final project report containing the pipeline schema, classification table, and IK sweep results.

---

## Getting Started

### 1. Prerequisites
Ensure you have ROS 2 Jazzy and Gazebo Harmonic installed, along with the `uv` package manager:
```bash
# Sourcing the environment will automatically activate the local .venv virtual environment
source setup_tp5.bash
```

### 2. Build the Workspace
To build the ROS 2 packages:
```bash
colcon build
source setup_tp5.bash
```

---

## Running the Simulation & Pipeline

### Terminal 1: Launch Gazebo Simulation
Starts the Gazebo Sim physics server, the GUI client, and spawns the 7-DOF arm:
```bash
source setup_tp5.bash
ros2 launch bcr_arm_gazebo bcr_arm.gazebo.launch.py
```

### Terminal 2: Run the Pick-and-Place Orchestration
```bash
source setup_tp5.bash
python3 src/bcr_arm/bcr_arm_gazebo/scripts/pick_and_place.py
```

---

## Standalone Tests
You can verify the mathematical modules (FK, IK, Trajectory, and Classifier) without launching Gazebo:
```bash
source setup_tp5.bash
python3 src/bcr_arm/bcr_arm_gazebo/scripts/test_pipeline_standalone.py
```
