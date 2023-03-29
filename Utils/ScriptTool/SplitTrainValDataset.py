# 根据标签进行图像数据集分类,分为验证集和训练集

import random
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def object_split_train_val_dataset(
    imgs_input_path: str,
    labels_input_path: str,
    output_path: str,
    train_sacle: float = 0.75,
) -> None:
    """
    根据图像对应标签个数进行采样,根据训练集比例随机分割训练集和验证集

    Args:
        imgs_input_path: 输入图片的路径
        labels_input_path: 输入图片对应的标签的路径
        output_path: 输出分割的数据集路径
        train_sacle: 训练集最后占整个的比例

    Returns:
        None
    """
    imgs_input_path = Path(imgs_input_path)
    labels_input_path = Path(labels_input_path)
    output_path = Path(output_path)
    output_path.joinpath("train").joinpath("images").mkdir(exist_ok=True, parents=True)
    output_path.joinpath("train").joinpath("labels").mkdir(exist_ok=True, parents=True)
    output_path.joinpath("val").joinpath("images").mkdir(exist_ok=True, parents=True)
    output_path.joinpath("val").joinpath("labels").mkdir(exist_ok=True, parents=True)

    logging.info("Getting the files path")
    labels_list = list(labels_input_path.glob("*.txt"))
    indices = random.sample(
        range(len(labels_list)), k=len(labels_list)
    )  # 随机采样,不会出现重复的情况
    train_indices = indices[: int(train_sacle * len(indices))]
    val_indices = indices[int(train_sacle * len(indices)) :]

    # 训练集复制
    for x in tqdm(train_indices, desc="Copying Train Dataset", unit="batchs"):
        src_label = labels_list[x]
        dst_label = (
            output_path.joinpath("train").joinpath("labels").joinpath(src_label.name)
        )
        src_image = imgs_input_path.joinpath(src_label.stem + ".jpg")  # ^ 图像格式后缀根据情况修改
        dst_image = (
            output_path.joinpath("train")
            .joinpath("images")
            .joinpath(src_label.stem + ".jpg")
        )
        shutil.copy(src_label, dst_label)
        shutil.copy(src_image, dst_image)

    # 验证集复制
    for x in tqdm(val_indices, desc="Copying Val Dataset", unit="batchs"):
        src_label = labels_list[x]
        dst_label = (
            output_path.joinpath("val").joinpath("labels").joinpath(src_label.name)
        )
        src_image = imgs_input_path.joinpath(src_label.stem + ".jpg")  # ^ 图像格式后缀根据情况修改
        dst_image = (
            output_path.joinpath("val")
            .joinpath("images")
            .joinpath(src_label.stem + ".jpg")
        )
        shutil.copy(src_label, dst_label)
        shutil.copy(src_image, dst_image)

    logging.info("All Finish! (*╹▽╹*),HaHa~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample images to train dataset and val dataset depending on bbox number!",
        epilog="For the sample balance!",
    )
    # parser.add_argument('-i', '--imgs_input_path', type=str, required=True, help='the path of source image', metavar='src_input_path')
    # parser.add_argument('-l', '--labels_input_path', type=str, default='./', help='the inputpath of labels!', metavar='labels_input_path')
    parser.add_argument(
        "-i",
        "--imgs_input_path",
        type=str,
        default="./images",
        help="the inputpath of source image",
        metavar="imgs_input_path",
    )  # For IDE
    parser.add_argument(
        "-l",
        "--labels_input_path",
        type=str,
        default="./labels",
        help="the inputpath of labels!",
        metavar="labels_input_path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./DataSet",
        help="the outpath of dataset dir",
        metavar="dataset_output_path",
    )
    parser.add_argument(
        "-ts",
        "--train_sacle",
        type=float,
        default=0.85,
        help="the scale of train dataset in all data!",
        metavar="train_sacle",
    )
    opt = parser.parse_args()

    object_split_train_val_dataset(
        opt.imgs_input_path, opt.labels_input_path, opt.output_path, opt.train_sacle
    )
