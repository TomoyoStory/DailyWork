# 语义分割颜色绘制,从而便于查看模型或者相关标签的情况

import os
import logging
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import Pool
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


IMG_FORMATS = [".png"]  # 当前仅保留了PNG图
NUM_THREADS = min(4, os.cpu_count())  # 进程数


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def _seg_color_and_save(args: tuple) -> None:
    """
    标签颜色的获取和保存,以下的代码根据实际的数据集的情况,适配性的修改。
    """
    mask_path, output_path, mix_src_path, mix_output_path = args
    mask = np.array(Image.open(mask_path))  # mask
    segmap = SegmentationMapsOnImage(
        mask, shape=mask.shape
    )  # SegmentationMapsOnImage原始draw颜色list查看源码
    color_mask = segmap.draw(size=mask.shape[:2])[0]
    dst_path = output_path.joinpath(mask_path.stem + ".jpg")  # jpg图
    Image.fromarray(color_mask).save(dst_path)

    if mix_src_path != None:  # mix src
        src_path = mix_src_path.joinpath(mask_path.stem + ".jpg")  #! 图像输入类别当前设定为固定jpg图
        src = np.array(Image.open(src_path))  # src
        color_mask_on_image = segmap.draw_on_image(src, alpha=0.5)[0]  # 透明度可根据情况修改
        mix_dst_path = mix_output_path.joinpath(mask_path.stem + ".jpg")
        Image.fromarray(color_mask_on_image).save(mix_dst_path)


def semantics_color(
    input_path: str, output_path: str, mix_src_path: str, mix_output_path: str
) -> None:
    """
    将原始的语义分割图(基本查看全是黑色)转化为由颜色表示的图像,并根据mix_src_path存在情况在原图上绘制

    Args:
        input_path: 输入的语义分割图的路径,该路径下为纯语义分割图
        output_path: 输出的由颜色表示的语义分割图
        mix_src_path: 如果要进行原图混合表示,输入的原图路径,注意,这里mix_src_path内部的数据名称前缀和input_path必须一致,且当前后缀为可源码修改的jpg格式
        mix_output_path: 输出的混合原图表示的语义分割图

    Returns:
        None
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    if mix_src_path != None:
        mix_src_path = Path(mix_src_path)
        mix_output_path = Path(mix_output_path)
        mix_output_path.mkdir(exist_ok=True)

    logging.info("Getting the files path")
    images_list = list(input_path.iterdir())
    logging.info("Filtering the files")
    images_list = [
        x for x in images_list if x.is_file() and x.suffix in IMG_FORMATS
    ]  # filter
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(
            pool.imap(
                _seg_color_and_save,
                zip(
                    images_list,
                    repeat(output_path),
                    repeat(mix_src_path),
                    repeat(mix_output_path),
                ),
            ),
            desc="Coloring the origin uint8 semantic segmentation image!",
            total=len(images_list),
            unit="imgs",
        )
        for i in pbar:
            pbar.update()

    logging.info("Color is beautiful, Right? Emmmmmmmm~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change uint8 Semantic segmentation image to color image for checking",
        epilog="Color means much! Really~",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="the path of uint8 Semantic segmentation image",
        metavar="seg_input_path",
    )
    # parser.add_argument('-i', '--input_path', type=str, default='./', help='the path of uint8 Semantic segmentation image', metavar='seg_input_path') # For IDE
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./labels_semantic_color",
        help="the path to get color image!",
        metavar="color_output_path",
    )
    parser.add_argument(
        "-mi",
        "--mix_src_path",
        type=str,
        default="./images",
        help="the path to src color image for fix image!",
        metavar="mix_input_path",
    )  # mix原图便于对比好看
    parser.add_argument(
        "-mo",
        "--mix_output_path",
        type=str,
        default="./labels_semantic_mix_color",
        help="the path to get mix color image!",
        metavar="mix_output_path",
    )  # mix图像输出路径
    opt = parser.parse_args()

    semantics_color(
        opt.input_path, opt.output_path, opt.mix_src_path, opt.mix_output_path
    )
