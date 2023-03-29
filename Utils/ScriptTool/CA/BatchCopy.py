# CA数据集根据文件名字批量复制对应的标签信息
# from_path必须为复制来自的路径,该路径必须为图像路径,而且图像路径的父目录,必须还有labels_lane, labels_obj, labels_semantic三个目录

import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def multi_task_batch_copy(
    copy_to_path: str,
    from_path: str,
    lane_dir_name: str = "labels_lane",
    obj_dir_name: str = "labels_obj",
    semantic_dir_name: str = "labels_semantic",
) -> None:
    """
    联合标注多任务数据标签进行文件复制,当前任务包括车道线识别、目标检测和语义识别

    Args:
        copy_to_path: 需要复制去的目录
        from_path: 复制的源目录,该目录下需要包括三个任务对应的标签
        lane_dir_name: 车道线标签数据的目录名称
        obj_dir_name: 目标检测标签数据的目录名称
        semantic_dir_name: 语义分割标签数据的目录名称

    Returns:
        None
    """
    copy2path = Path(copy_to_path)
    copy_from_path = Path(from_path)

    copy2path.parent.joinpath(lane_dir_name).mkdir(exist_ok=True)
    copy2path.parent.joinpath(obj_dir_name).mkdir(exist_ok=True)
    copy2path.parent.joinpath(semantic_dir_name).mkdir(exist_ok=True)
    for file in tqdm(
        list(copy2path.iterdir()), desc="Copying the files", unit="batchs"
    ):
        if file.is_file():
            absolute_path = str(copy_from_path.resolve().joinpath(file.stem))
            lane_label_path = (
                lane_dir_name.join(absolute_path.rsplit("images", 1)) + ".json"
            )
            object_label_path = (
                obj_dir_name.join(absolute_path.rsplit("images", 1)) + ".txt"
            )
            mask_label_path = (
                semantic_dir_name.join(absolute_path.rsplit("images", 1)) + ".png"
            )
            shutil.copy(lane_label_path, copy2path.parent.joinpath(lane_dir_name))
            shutil.copy(object_label_path, copy2path.parent.joinpath(obj_dir_name))
            shutil.copy(mask_label_path, copy2path.parent.joinpath(semantic_dir_name))
    logging.info("All Finish! (*╹▽╹*),HaHa~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CA Dataset Tool, Copy the label file corresponding file path ",
        epilog="MultiTask is not easy!",
    )
    parser.add_argument(
        "-p",
        "--copy_to_path",
        type=str,
        default="./path1/images",
        required=True,
        help="copy path!",
        metavar="file_path",
    )
    parser.add_argument(
        "-f",
        "--from_path",
        type=str,
        default="./path2/images",
        required=True,
        help="from path! this path must be images path and dir tree must be same!",
        metavar="from_path",
    )
    # parser.add_argument('-p', '--copy_to_path', type=str, default='./path1/images', help='copy path!', metavar='file_path') # For IDE
    # parser.add_argument('-f', '--from_path', type=str, default='./path2/images', help='from path! this path must be images path and dir tree must be same!', metavar='from_path')
    opt = parser.parse_args()

    multi_task_batch_copy(opt.copy_to_path, opt.from_path)
