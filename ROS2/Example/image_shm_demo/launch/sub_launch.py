from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # pkg_dir = get_package_share_directory("image_shm_demo")

    # Fast DDS SHM 配置文件路径
    # fastdds_shm_xml = PathJoinSubstitution([pkg_dir, "config", "fastdds_shm_profile.xml"])

    # 1. 全局环境变量：启用Fast DDS Data-Sharing
    env_setup = [
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("RMW_FASTRTPS_USE_QOS_FROM_XML", "1"),
        # SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", fastdds_shm_xml),
        SetEnvironmentVariable("ROS_DISABLE_LOANED_MESSAGES", "0"), #! 订阅端必须设置这个环境变量才能启动共享内存接受,ROS2官网上有说明
    ]

    # ========== 进程2：图像订阅容器（独立进程，跨进程SHM传输） ==========
    sub_container = ComposableNodeContainer(
        name="sub_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=["--ros-args", "--disable-rosout-logs"],  #! 注意,这里必须添加--ros-args，不然是不起作用的
        # arguments=["--ros-args", "--disable-rosout-logs","--disable-stdout-logs"],
        composable_node_descriptions=[
            ComposableNode(
                package="image_shm_demo",
                plugin="image_shm_demo::VecSubNode",
                name="image_sub_node",
                extra_arguments=[{"use_intra_process_comms": True}], #! 可以激活,节点内的pub使用进程内通讯,sub使用进程间共享内存
            )
        ],
        output="screen",
    )

    return LaunchDescription(env_setup + [sub_container])