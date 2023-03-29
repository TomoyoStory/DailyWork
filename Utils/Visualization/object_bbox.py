# YOLO格式目标检测框绘制,便于查看框的情况,此外,也便于验证标注的情况。
# 该部分代码参考YOLOv5官网源码获取

import cv2
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            "FF9D97",
            "FF3838",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class BboxAnnotator:
    def __init__(
        self,
        im: np.array,
        line_width: int = None,
        font_size: int = None,
        font: str = "Arial.ttf",
        pil: bool = False,
    ):
        """
        将原始的语义分割图(基本查看全是黑色)转化为由颜色表示的图像,并根据mix_src_path存在情况在原图上绘制

        Args:
            im: 输入im应该是一个np.array代表的图像
            line_width: 绘制的线宽,如果没有设置,默认根据图像大小自适应选择
            font_size: 字体的大小,如果没有设置,默认根据图像大小自适应选择
            font: 当前为了美观,采用Arial.ttf这个true type font
            pil: 是否使用pil进行绘制

        Returns:
            None
        """
        self.pil = pil
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            font_path = str(Path(__file__).resolve().parent.joinpath(font))
            self.font = ImageFont.truetype(
                font_path, font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
            )
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(
        self,
        box: Union[np.array, list],
        label: str = "",
        color: tuple = (128, 128, 128),
        txt_color: tuple = (255, 255, 255),
    ) -> None:
        """
        绘制xyxy格式的标签图在指定的图像上

        Args:
            box: 框,格式为xyxy
            label: 标签的名称
            color: 指定文字背景底色的颜色
            txt_color: 指定文字的颜色

        Returns:
            None
        """
        # Add one xyxy box to image with label
        if self.pil:
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box 超过了直接放框下面
                self.draw.rectangle(
                    [
                        box[0],
                        box[1] - h if outside else box[1],
                        box[0] + w + 1,
                        box[1] + 1 if outside else box[1] + h + 1,
                    ],
                    fill=color,
                )
                self.draw.text(
                    (box[0], box[1]), label, fill=txt_color, font=self.font, anchor="ls"
                )  # for PIL>8.0
                # self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font) # for earlier versions of Pillow
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(
                self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[
                    0
                ]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    # ^ 该函数仅做了封装,具体接口参数需根据使用情况修改
    def rectangle_pil(
        self,
        box: Union[np.array, list],
        fill: Union[tuple, None] = None,
        outline: Union[tuple, None] = None,
        width: int = 1,
    ):
        """
        使用Pillow库进行框的绘制

        Args:
            box: 框,格式为xyxy
            fill: 填充使用的颜色,类似于(128, 128, 128),None表示不填充
            outline: 外轮廓线使用的颜色,类似于(128, 128, 128),None表示不画外框
            width: 外轮廓线的宽度,像素单位

        Returns:
            None
        """
        if self.pil:
            self.draw.rectangle(box, fill, outline, width)
        else:
            raise Exception('BboxAnnotator Class do not use "pil=False"')

    # ^ 该函数仅做了封装,具体接口参数需根据使用情况修改
    def rectangle_cv(
        self, box: Union[np.array, list], color: tuple = (128, 128, 128)
    ) -> None:
        """
        使用OpenCv库进行框的绘制

        Args:
            box: 框,格式为xyxy
            color: 填充使用的颜色,类似于(128, 128, 128)

        Returns:
            None
        """
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

    # ^ 该函数仅做了封装,具体接口参数需根据使用情况修改
    def text_pil(
        self, xy: Union[np.array, list], text: str, txt_color: tuple = (255, 255, 255)
    ) -> None:
        """
        使用Pillow库进行文本的绘制

        Args:
            xy: 绘制锚点的位置(Pillow 8.0版本加入了锚框位置选择可查看)
            text: 要绘制的文本
            txt_color: 文本绘制的颜色

        Returns:
            None
        """
        if self.pil:
            self.draw.text(
                (xy[0], xy[1]), text, fill=txt_color, font=self.font, anchor="ls"
            )  # for PIL>8.0
            # w, h = self.font.getsize(text)  # text width, height
            # self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font) # for earlier versions of Pillow
        else:
            raise Exception('BboxAnnotator Class do not use "pil=False"')

    def text_cv(
        self,
        box: Union[np.array, list],
        text: str,
        color: tuple = (128, 128, 128),
        txt_color: tuple = (255, 255, 255),
    ) -> None:
        """
        使用OpenCv库进行加入背景颜色框的文本的绘制

        Args:
            box: 框,xyxy
            text: 要绘制的文本
            color:绘制文本对应的背景颜色
            txt_color: 文本绘制的颜色

        Returns:
            None
        """
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        tf = max(self.lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(text, 0, fontScale=self.lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            self.im,
            text,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            self.lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    def result(self):
        """
        返回标签的np.array结果

        Returns:
            np.array
        """
        return np.asarray(self.im)


def bbox_color(image_input_path: str, label_input_path: str, output_path: str):
    """
    根据标签文件和原始图片绘制标注文件,便于整体的查看。

    Args:
        image_input_path: 源图像的输入路径
        label_input_path: 标签图像的输入路径
        color:绘制文本对应的背景颜色
        txt_color: 文本绘制的颜色

    Returns:
        None
    """
    image_input_path = Path(image_input_path)
    label_input_path = Path(label_input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    colors = Colors()

    logging.info(f"Reading the labels from {str(label_input_path)}.")
    # TODO 多进程加速绘制流程
    labels = [x for x in list(label_input_path.iterdir()) if x.suffix in ".txt"]
    for label in tqdm(
        labels, desc="Drawing the bbox to images and save the images", unit="batchs"
    ):
        image_path = image_input_path.joinpath(label.stem + ".jpg")
        image = np.array(Image.open(image_path))  # image
        height, width = image.shape[0], image.shape[1]
        annotator = BboxAnnotator(image, pil=True)
        with open(label, "r", encoding="utf-8") as f:
            for cxywh in f.readlines():
                cxywh = cxywh.strip().split()
                c = cxywh[0]
                x1 = (float(cxywh[1]) - float(cxywh[3]) / 2) * width
                y1 = (float(cxywh[2]) - float(cxywh[4]) / 2) * height
                x2 = (float(cxywh[1]) + float(cxywh[3]) / 2) * width
                y2 = (float(cxywh[2]) + float(cxywh[4]) / 2) * height
                bbox = (x1, y1, x2, y2)
                annotator.box_label(bbox, str(c), colors(c, True))
        im0 = annotator.result()
        Image.fromarray(im0).save(
            output_path.joinpath(label.stem + ".jpg")
        )  # 根据情况选择保存的后缀
    logging.info("All Finish! (*￣︶￣) ,233~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the bounding box to image for checking!",
        epilog="Color means much! Really~",
    )
    parser.add_argument(
        "-i",
        "--image_input_path",
        type=str,
        required=True,
        help="the path of image",
        metavar="image_input_path",
    )  # For IDE
    parser.add_argument(
        "-l",
        "--label_input_path",
        type=str,
        required=True,
        help="the path of label! the format is YOLO(cxywh)",
        metavar="label_input_path",
    )
    # parser.add_argument('-i', '--image_input_path', type=str, default='./images', help='the path of image', metavar='image_input_path') # For IDE
    # parser.add_argument('-l', '--label_input_path', type=str, default='./labels', help='the path of label! the format is YOLO(cxywh)', metavar='label_input_path')
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./bbox_images",
        help="the path to get bounding box image!",
        metavar="bbox_output_path",
    )
    opt = parser.parse_args()

    bbox_color(opt.image_input_path, opt.label_input_path, opt.output_path)
