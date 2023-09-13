# 图像坐标系下的相关可视化验证测试脚本
# 注意#!部分需根据实际情况修改

import cv2
import logging
import argparse
import numpy as np
import open3d as o3d

from pathlib import Path

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-sensor visualization tool for cameras",
        epilog="Autonomous driving may be a false proposition! Really~",
    )
    parser.add_argument(
        "-im",
        "--image_input_file",
        type=str,
        default="./example/calibration_raw.png",
        help="the path of distort raw image",
        metavar="{distort raw input path}",
    )
    parser.add_argument(
        "-p",
        "--pcd_input_file",
        type=str,
        default="./example/calibration_raw.pcd",
        help="the lidar pointcloud",
        metavar="{PCL pcd file}",
    )
    parser.add_argument(
        "-in",
        "--intrinsics_input_file",
        type=str,
        default="./example/intrinsics.yml",
        help="camera intrinsics parameters(opencv style, include distortion parameters and intrinsic matrix)",
        metavar="{camera intrinsics parameters(opencv style)}",
    )
    parser.add_argument(
        "-ex",
        "--extrinsics_input_file",
        type=str,
        default="./example/lidar2img_extrinsics.yml",
        help="camera extrinsics parameters(opencv style, lidar frame to camera frame)",
        metavar="{camera extrinsics parameters(opencv style)}",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./example/result",
        help="output dir path, include undistort image, lidar or radar point project image",
        metavar="{output dir include undistort image、lidar point project 、radar point project image}",
    )
    parser.add_argument(
        "-d",
        "--depth_filter",
        type=float,
        default=40,
        help="pointcloud in camera frame depth filter threshold",
        metavar="{filter far pointcloud for better visual image!}",
    )  # 过滤掉距离图像坐标系一定距离的激光点,便于可视化,较远的点会对图像的点云映射可视化产生干扰

    opt = parser.parse_args()

    # 读取相机内参文件,获取内参矩阵和畸变系数(保存格式使用标准opencv的yml格式)
    logging.info("Reading the intrinsics parameters!")
    intrinsics = cv2.FileStorage(opt.intrinsics_input_file, cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()  # ^ 3x3 相机标定完成的内参矩阵,具体K名称可能和存储名称相关
    D = intrinsics.getNode("D").mat()  # ^ 5x1 相机标定完成的畸变系数,具体D名称可能和存储名称相关

    # 读取激光雷达到相机的外参矩阵
    logging.info("Reading the extrinsics parameters!")
    extrinsics = cv2.FileStorage(opt.extrinsics_input_file, cv2.FILE_STORAGE_READ)
    lidar2img = extrinsics.getNode(
        "lidar2img"
    ).mat()  # ^ 4x4 激光雷达到相机坐标系转换矩阵,具体lidar2img名称可能和存储名称相关

    # 读取图片和点云数据，并对原始图片进行矫正
    logging.info("Reading the image and undistort image!")
    image = cv2.imread(opt.image_input_file)
    image = cv2.undistort(image, K, D)
    output_dir = Path(opt.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)  # 创建结果目录
    logging.info("Saving the undistort image!")
    cv2.imwrite(
        str(output_dir.joinpath(Path(opt.image_input_file).stem + "_undistort.png")),
        image,
    )  # 保存矫正后的原始图像
    logging.info("Reading the pointcloud!")
    point_cloud = o3d.io.read_point_cloud(opt.pcd_input_file)

    # 将点云从激光雷达坐标系转换到相机坐标系
    logging.info("Changing the pointcloud from lidar frame to camera frame!")
    points = np.asarray(point_cloud.points)
    points_lidar_frame = np.hstack(
        (points, np.ones((points.shape[0], 1)))
    ).T  # 4xN 激光雷达点云,最后一个维度为齐次项
    points_camera_frame = np.dot(lidar2img, points_lidar_frame)[:3, :]  # 摄像头坐标系下的点云
    points_filter = []  # ^ 过滤较深的点云
    for point in points_camera_frame.T:
        if point[2] < opt.depth_filter and point[2] > 0:
            points_filter.append(point)
    points_filter = np.array(points_filter).T

    # 将点云从相机坐标系投影到图像平面
    logging.info("Changing the pointcloud from camera frame to image frame!")
    points_image_frame = np.dot(K, points_filter)
    points_image_frame_normalized = (
        points_image_frame[:2, :] / points_image_frame[2, :]
    )  # 2xN 根据单应性矩阵映射原理归一化深度

    # 根据深度配置图像上点云的颜色
    color_map = 255 - 255 * points_filter[2, :] / opt.depth_filter
    color_map = color_map.astype(np.uint8)
    color_map = color_map[np.newaxis, :, np.newaxis]
    color_map = cv2.applyColorMap(
        color_map, cv2.COLORMAP_JET
    )  # ^ 这里可以参考Opencv接口选择适宜的颜色带
    color_map = np.squeeze(color_map, axis=0)

    # 根据映射的图像范围进行点云着色绘制
    logging.info("Color the pointcloud in image frame!")
    for i, point in enumerate(points_image_frame_normalized.T):
        if (
            point[0] >= 0
            and point[0] < image.shape[1]
            and point[1] >= 0
            and point[1] < image.shape[0]
        ):
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                1,
                tuple([int(x) for x in color_map[i]]),
                -1,
            )

    # 毫米波雷达定位位置显示
    logging.info("Changing the radar obstacle from lidar frame to image frame!")
    obstacle_lidar_frame = [
        -2.48,
        -10.11,
        -2.30,
        1,
    ]  #! 4x1 毫米波雷达在激光雷达坐标系下检测到的障碍物坐标，其中第四个维度为齐次项。障碍物坐标根据实际情况自定选择确认,格式为[x,y,z,1],这里为例程对应的毫米波雷达检测到的障碍物
    logging.info(f"The radar obstacle coordinate is : {obstacle_lidar_frame}")
    radar_lidar_frame = np.array(obstacle_lidar_frame)[:, np.newaxis]
    radar_camera_frame = np.dot(lidar2img, radar_lidar_frame)[:3, :]  # 3x1
    radar_image_frame = np.dot(K, radar_camera_frame)  # 2x1
    radar_image_frame_normalized = (
        radar_image_frame[:2, :] / radar_image_frame[2, :]
    )  # 2x1 根据单应性矩阵映射原理归一化深度
    cv2.circle(
        image,
        (
            int(radar_image_frame_normalized[0][0]),
            int(radar_image_frame_normalized[1][0]),
        ),
        5,
        (0, 255, 0),
        -1,
    )  # ^ 毫米波雷达绘制的点的颜色和大小根据个人喜好自行选择

    # 地面障碍物反向映射测试
    logging.info("Inver Testing the image obstacle to lidar frame!")
    k_inv = np.linalg.inv(K)  # 单应性矩阵逆矩阵
    image_pix = np.array(
        [radar_image_frame_normalized[0][0], radar_image_frame_normalized[1][0], 1]
    )[
        :, np.newaxis
    ]  # 加入齐次项
    image_image_frame = k_inv @ image_pix  # 在原始相机坐标系下的投射极线
    camera_height = 1.7431326371803877  #! 相机高度,标定的时候求解得到
    depth = camera_height / image_image_frame[1, 0]  # 对应地面障碍物相机坐标系下的深度
    image_image_frame = depth * image_image_frame  # 在原始相机坐标系下的坐标
    image_frame = np.array(
        [image_image_frame[0, 0], image_image_frame[1, 0], image_image_frame[2, 0], 1]
    )[
        :, np.newaxis
    ]  # 加入齐次项
    img2lidar = np.linalg.inv(lidar2img)  # 相机坐标系到激光雷达坐标系的转换矩阵
    image_lidar_frame = img2lidar @ image_frame
    image_lidar_frame = np.squeeze(image_lidar_frame, axis=1)[:3]
    logging.info(
        f"the image obstacle's coordinate in lidar frame is : {image_lidar_frame}"
    )

    # 保存最后的图像
    cv2.imwrite(
        str(output_dir.joinpath(Path(opt.image_input_file).stem + "_result.png")), image
    )
    logging.info("Is that all rihgt? Please Check the result image!")
