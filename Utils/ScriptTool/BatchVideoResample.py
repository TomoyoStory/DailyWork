# 递归文件目录批量读取视频数据，并针对视频数据进行图像数据重采样和批量文件保存

import os
import cv2
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

VIDEO_FORMATS = ["mp4", "avi"]  # 支持的视频格式后缀名


def batch_resample_from_video(
    video_dir: str, output_path: str, sample_rate: int
) -> None:
    """
    递归读取video_dir目录下的所有视频,根据sample_rate的采样率进行数据采样,生成序列图片

    Args:
        video_dir: 输入包含视频文件的目录地址
        output_path: 保持输入目录格式的图像输出目录
        sample_rate: 批量命名的起始位置

    Returns:
        None
    """
    video_path = Path(video_dir)
    output_path = Path(output_path)

    # 视频文件递归搜寻
    logging.info("Searching the video files")
    files = list(video_path.glob("*"))
    files = [
        Path(x) for x in files if str(x).split(".")[-1].lower() in VIDEO_FORMATS
    ]  # 获取视频文件图像路径

    # 图像重采样
    for x in tqdm(files, desc="Resampling the video!", unit="files"):
        relative_path = x.relative_to(video_path)  # 相对路径
        absolute_path = output_path.joinpath(relative_path).parent.joinpath(
            x.stem
        )  # 绝对图像保存地址
        absolute_path.mkdir(exist_ok=True, parents=True)  # 创建目录
        capture = cv2.VideoCapture(str(x))
        if not capture.isOpened():
            raise Exception(f"ERROR: {str(x)} Can not been opened")

        count = 0  # 初始计数
        while True:
            ret, frame = capture.read()
            if not ret:
                logging.info(f"{str(x)} has run out!")
                break
            if count % sample_rate == 0:
                image_name = absolute_path.joinpath(
                    str(count).zfill(8) + ".jpg"
                )  # ^ 位数根据实际的情况修改
                cv2.imwrite(str(image_name), frame)
            count = count + 1
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read the video file recursively and Resample the video image!",
        epilog="Yesterday was the Chinese Valentine's Day, My Girl Fridend is cute!",
    )
    # parser.add_argument('-p', '--path', type=str, default='./', required=True, help='the directory path to get the video files!', metavar='video_path')
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=r"D:\DATA\2022-8-2红绿灯数据采集",
        help="the directory path to get the video files!",
        metavar="video_path",
    )  # For IDE
    # parser.add_argument('-s', '--sample_rate', type=int, default=5, required=True, help='video sampling rate', metavar='sample_rate')
    parser.add_argument(
        "-s",
        "--sample_rate",
        type=int,
        default=5,
        help="video sampling rate",
        metavar="sample_rate",
    )  # For IDE
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./video_images",
        help="The output path of resampled image!",
        metavar="output_images_path",
    )
    opt = parser.parse_args()

    batch_resample_from_video(opt.path, opt.output_path, opt.sample_rate)
