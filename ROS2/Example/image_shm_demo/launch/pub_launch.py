from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory("image_shm_demo")

    # Fast DDS SHM 配置文件路径
    fastdds_shm_xml = PathJoinSubstitution([pkg_dir, "config", "fastdds_shm_profile.xml"])

    # 1. 全局环境变量：启用Fast DDS Data-Sharing
    env_setup = [
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("RMW_FASTRTPS_USE_QOS_FROM_XML", "1"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", fastdds_shm_xml),
        # SetEnvironmentVariable("ROS_DISABLE_LOANED_MESSAGES", "0"),
    ]

    # ========== 进程1：图像发布容器 ==========
    pub_container = ComposableNodeContainer(
        name="pub_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=["--ros-args", "--disable-rosout-logs"],  #! 注意,这里必须添加--ros-args，不然是不起作用的
        # arguments=["--ros-args", "--disable-rosout-logs","--disable-stdout-logs"],
        composable_node_descriptions=[
            ComposableNode(
                package="image_shm_demo",
                plugin="image_shm_demo::VecPubNode",
                name="image_pub_node",
                #! 当前设置pub端不能开启进程内通讯,不然不能进行data_sharing,这里对于pub只能2选1
                # extra_arguments=[{"use_intra_process_comms": True}],
            )
        ],
        output="screen",
    )

    return LaunchDescription(env_setup + [pub_container])