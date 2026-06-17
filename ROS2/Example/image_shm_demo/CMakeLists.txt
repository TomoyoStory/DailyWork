cmake_minimum_required(VERSION 3.8)
project(image_shm_demo)

#! 以下整个项目使用的是基于ament_cmake_auto的极简版本的模板设置

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# find dependencies #! 自动化发现依赖!
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 包内接口生成
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImageData.msg"
  DEPENDENCIES builtin_interfaces)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

# 图像发布组件
ament_auto_add_library(image_pub_component SHARED src/image_pub_component.cpp)
target_include_directories(image_pub_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/image_shm_demo>
    $<INSTALL_INTERFACE:include>)
rclcpp_components_register_node(image_pub_component
                                PLUGIN "image_shm_demo::VecPubNode"
                                EXECUTABLE shm_pub_node)
target_link_libraries(image_pub_component "${cpp_typesupport_target}")

# 图像订阅组件
ament_auto_add_library(image_sub_component SHARED src/image_sub_component.cpp)
target_include_directories(image_sub_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/image_shm_demo>
    $<INSTALL_INTERFACE:include>)
rclcpp_components_register_node(image_sub_component
                                PLUGIN "image_shm_demo::VecSubNode"
                                EXECUTABLE shm_sub_node)
target_link_libraries(image_sub_component "${cpp_typesupport_target}")


#-----------------------以下部分主要是为了说明构建多个组件节点到一个.so文件中----------------------------
#!!!! 冗余
#!!!! 这里是冗余的一次性进行多个组件节点注册到一个.so文件的说明
#!!!! 一定要注意rclcpp_components_register_nodes和rclcpp_components_register_node的区别,特别是EXECUTABLE参数作用
# ament_auto_add_library(image_pub_sub_component SHARED src/image_pub_component.cpp src/image_sub_component.cpp)
# target_include_directories(image_pub_sub_component PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/image_shm_demo>
#     $<INSTALL_INTERFACE:include>)
# rclcpp_components_register_nodes(image_pub_sub_component
#                                  "image_shm_demo::VecPubNode"
#                                  "image_shm_demo::VecSubNode")
# target_link_libraries(image_pub_sub_component "${cpp_typesupport_target}")

# # 额外安装所有组件在一个.so文件中的库文件
# install(TARGETS image_pub_sub_component
#         EXPORT export_image_pub_sub_component
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION bin)
#-----------------------以上部分主要是为了说明构建多个组件节点到一个.so文件中----------------------------


# 安装
install(TARGETS image_pub_component
        EXPORT export_image_pub_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS image_sub_component
        EXPORT export_image_sub_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

#! 使用了ament_auto_package()之后,对应依赖的头文件会自动安装
#! 这里的visibility_control.h文件会安装到install/image_shm_demo/include/image_shm_demo/目录
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION include)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

# ament_export_targets(export_image_pub_component HAS_LIBRARY_TARGET)
# ament_export_targets(export_image_sub_component HAS_LIBRARY_TARGET)

ament_auto_package()
