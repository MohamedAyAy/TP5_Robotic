#!/bin/bash
# TP5 Workspace Setup
# ===================
# Source this file ONCE in every terminal before running any ros2 command:
#
#   source ~/bcr_ws1/setup_tp5.bash
#
# After sourcing, all these commands will work:
#   ros2 run bcr_arm_gazebo ik_solver.py
#   ros2 run bcr_arm_gazebo test_pipeline_standalone.py
#   ros2 run bcr_arm_gazebo pick_and_place.py
#   ros2 launch bcr_arm_gazebo bcr_arm.gazebo.launch.py

# 1. Source ROS 2 Jazzy base
source /opt/ros/jazzy/setup.bash

# 2. Source this workspace's install directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/install/setup.bash"

# 3. Add ROS 2 packages to PYTHONPATH (enables ROS 2 import inside uv venv)
export PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH"

# 4. Activate uv virtual environment
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "⚡  UV Virtual Environment (.venv) activated."
fi

# 5. Confirm packages are found
echo "✅  ROS 2 Jazzy + bcr_ws1 sourced."
echo "    Available bcr packages:"
ros2 pkg list | grep bcr | sed 's/^/      /'

echo ""
echo "Quick test commands:"
echo "  ros2 run bcr_arm_gazebo ik_solver.py"
echo "  ros2 run bcr_arm_gazebo test_pipeline_standalone.py"
echo "  ros2 launch bcr_arm_gazebo bcr_arm.gazebo.launch.py"
