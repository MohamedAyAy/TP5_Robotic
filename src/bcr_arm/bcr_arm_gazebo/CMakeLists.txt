cmake_minimum_required(VERSION 3.8)
project(bcr_arm_gazebo)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config worlds scripts
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/test_arm_movement.py
  scripts/forward_kinematics.py
  scripts/kinematics_nn.py
  scripts/predict_fk.py
  scripts/perception.py
  scripts/segmentation.py
  scripts/classifier.py
  scripts/ik_solver.py
  scripts/trajectory.py
  scripts/pick_and_place.py
  scripts/dgcnn_model.py
  scripts/bcr_kinematics.py
  scripts/download_weights.py
  scripts/test_pipeline_standalone.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
