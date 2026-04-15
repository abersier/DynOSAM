from dynosam_ros.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from dynosam_ros.launch_utils import get_default_dynosam_params_path


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path()),
        DeclareLaunchArgument("v", default_value="30"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        DeclareLaunchArgument("camera_info_topic", default_value="/robot_1/rgbd_camera/camera_info"),
        DeclareLaunchArgument("rgb_cam_topic", default_value="/robot_1/rgbd_camera/image"),
        DeclareLaunchArgument("depth_cam_topic", default_value="/robot_1/rgbd_camera/depth_image"),
        DeclareLaunchArgument("mask_cam_topic", default_value="/robot_1/mask"),
        DeclareLaunchArgument("rescale_width", default_value="640", description="Image width to rescale to"),
        DeclareLaunchArgument("rescale_height", default_value="480", description="Image height to rescale to"),
        DeclareLaunchArgument("depth_scale", default_value="1.0", description="Scale factor to convert depth to metres (1.0 for Gazebo float32, 0.001 for mm sensors)"),
        DeclareLaunchArgument("frame_stride", default_value="1", description="Process only every Nth frame (1=all, 3=every 3rd). Set to match camera Hz / pipeline throughput."),
        DeclareLaunchArgument("ego_namespace", default_value="robot_1", description="Robot namespace used to look up map -> <ego_ns>/optical_frame for world frame init"),

        # Broadcasts a one-time static TF map -> world by looking up map -> <ego_ns>/optical_frame at startup.
        # This aligns DynOSAM's 'world' frame with Gazebo's 'map' frame for RViz overlay with DynORecon.
        Node(
            package="dyno_mpc_sim",
            executable="dynosam_world_init.py",
            name="dynosam_world_init",
            output="screen",
            parameters=[
                {"ego_namespace": LaunchConfiguration("ego_namespace")},
                {"world_frame_id": "world"},
            ],
        ),

        DynosamNode(
                package="dynosam_ros",
                executable="dynosam_node",
                output="screen",
                parameters=[
                    {"params_folder_path": LaunchConfiguration("params_path")},
                    {"rescale_width": LaunchConfiguration("rescale_width")},
                    {"rescale_height": LaunchConfiguration("rescale_height")},
                    {"online": True},
                    {"input_image_mode": 3}, # Corresponds with InputImageMode::RGBDM
                    {"depth_scale": LaunchConfiguration("depth_scale")},
                    {"frame_stride": LaunchConfiguration("frame_stride")}
                ],
                remappings=[
                    ("dataprovider/camera/camera_info",  LaunchConfiguration("camera_info_topic")),
                    ("image/rgb",  LaunchConfiguration("rgb_cam_topic")),
                    ("image/depth",  LaunchConfiguration("depth_cam_topic")),
                    ("image/mask",  LaunchConfiguration("mask_cam_topic"))
                ]
            ),
        ])
