import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

# At the top of generate_launch_description, get the path to your world file
pkg_gazebo = FindPackageShare("bcr_arm_gazebo")
world_path = PathJoinSubstitution([pkg_gazebo, "worlds", "pick_scene.world"])

# Then in your gazebo IncludeLaunchDescription:

def generate_launch_description():

    pkg_gazebo      = FindPackageShare("bcr_arm_gazebo")
    pkg_description = FindPackageShare("bcr_arm_description")

    # ------------------------------------------------------------------
    # FIX 1: Tell Gazebo where to find package:// mesh resources.
    # gz-sim8 does NOT translate package:// URIs on its own — we must
    # add the ROS install/share root to GZ_SIM_RESOURCE_PATH so Gazebo
    # can resolve  model://bcr_arm_description/meshes/...
    # ------------------------------------------------------------------
    ros_share = os.path.join(
        get_package_share_directory("bcr_arm_description"),  # .../share/bcr_arm_description
        "..",                                                # .../share
    )
    gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
            ":" + os.path.normpath(ros_share),
        ],
    )

    # ------------------------------------------------------------------
    # Generate URDF via xacro
    # ------------------------------------------------------------------
    robot_description_command = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([pkg_description, "urdf", "robots", "bcr_arm.urdf.xacro"]),
        " ",
        "use_gazebo:=true",
        " ",
        # Note: depth camera is always included in bcr_arm.urdf.xacro unconditionally
        " ",
        "ros2_controllers_path:=",
        PathJoinSubstitution([pkg_gazebo, "config", "ros2_controllers.yaml"]),
    ])

    robot_description = ParameterValue(robot_description_command, value_type=str)

    # ------------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    # ------------------------------------------------------------------
    # Gazebo Sim
    # ------------------------------------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={"gz_args": ["-r -v 4 ", world_path]}.items(),
    )

    # ------------------------------------------------------------------
    # FIX 2: Spawn robot via /robot_description topic, NOT via -string.
    # Passing a Command() substitution object directly to -string does
    # not work in gz-sim8 — the substitution never resolves correctly.
    # The robot_state_publisher already publishes on /robot_description,
    # so we just point the spawner there.
    # ------------------------------------------------------------------
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "/robot_description",
            "-name",  "bcr_arm",
            "-z",     "0.0",
        ],
    )

    # ------------------------------------------------------------------
    # Bridge ROS 2 <-> Gazebo (clock only)
    # ------------------------------------------------------------------
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # ------------------------------------------------------------------
    # FIX 3: Controllers — add a TimerAction delay after spawning so the
    # gz_ros2_control plugin has time to register the controller manager.
    # Using only OnProcessExit(spawn_entity) is too fast on gz-sim8.
    # ------------------------------------------------------------------
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    joint_trajectory_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # Wait 3 s after spawn_entity exits before starting joint_state_broadcaster
    delay_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[joint_state_broadcaster],
                )
            ],
        )
    )

    # Wait for joint_state_broadcaster to finish before starting trajectory controller
    delay_jtc = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[joint_trajectory_controller],
        )
    )
    camera_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    arguments=[
        "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
        "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
        "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
    ],
)

    return LaunchDescription([
        gz_resource_path,       # must be first so Gazebo inherits it
        robot_state_publisher,
        gazebo,
        spawn_entity,
        gz_bridge,
        camera_bridge,
        delay_jsb,
        delay_jtc,
        
    ])
