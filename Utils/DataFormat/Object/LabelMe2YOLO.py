# VOC的xml格式转换为YOLO的xywh格式

import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


# 目标检测类别 #^ 这里根据实际情况修改
OBJECT_DICT = {
    # Name            ID          TrainID           TypeID
    "personal": {"id": 0, "train_id": 0, "type_id": 0},  # 行人
    "police": {"id": 1, "train_id": 1, "type_id": 0},  # 警察
}


def labelme2yolo(input_path: str, output_path: str, count_output_file: str) -> None:
    """
    将LabelMe的json格式数据转换为YOLOv5的xywh格式

    Args:
        input_path: 输入json标签的路径
        output_path: 输出txt标签的路径
        count_output_file: 统计文件的绝对路径

    Returns:
        None
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # 类别统计
    object_dict = {}
    for key in OBJECT_DICT.keys():
        object_dict[key] = 0
    object_dict["other"] = 0  # 其他第三方类别

    logging.info("Getting the json labels")
    files = list(input_path.glob("*.json"))  # 仅读取json

    for label_file in tqdm(
        files, desc="Changing LabelMe json format to YOLO format!", unit="jsons"
    ):
        with open(label_file, "r", encoding="utf-8") as f:
            json_str = json.load(f)
        width = json_str["imageWidth"]
        height = json_str["imageHeight"]
        label_str = ""
        for x in json_str["shapes"]:
            label_name = x["label"]
            train_id = OBJECT_DICT[label_name]["train_id"]
            # 标签名称转换和数量统计 #^ 这里根据实际情况修改
            try:
                object_dict[label_name] += 1
            except:
                object_dict["other"] += 1

            if train_id != -1:
                x_cnetral = (x["points"][0][0] + x["points"][1][0]) / (2 * width)
                y_central = (x["points"][0][1] + x["points"][1][1]) / (2 * height)
                w = abs((x["points"][1][0] - x["points"][0][0])) / width
                h = abs((x["points"][1][1] - x["points"][0][1])) / height
                label_str = (
                    label_str
                    + str(train_id)
                    + " "
                    + " ".join("%.6f" % i for i in [x_cnetral, y_central, w, h])
                    + "\n"
                )

        with open(
            output_path.joinpath(label_file.stem + ".txt"), "w", encoding="utf-8"
        ) as f:
            f.write(label_str)

    # 类别个数统计
    with open(count_output_file, "w", encoding="utf-8") as f:
        for key in object_dict.keys():
            f.write(key + ":" + str(object_dict[key]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change LabelMe json dataset format to yolo dataset format! Please Change the OBJECT_DICT in PYTHON FILE for your class!",
        epilog="Hello! AI dataset! Hummmm....",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="the file input path of LabelMe json format to yolov5 format!",
        metavar="labelme_json_input_path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./yolo_format",
        help="the file output path of LabelMe json format to yolov5 format!",
        metavar="lableme_json_output_path",
    )
    # 便于在IDE中运行
    # parser.add_argument('-i', '--input_path', type=str, default='./', help='the file input path of LabelMe json format to yolov5 format!', metavar='labelme_json_input_path')
    # parser.add_argument('-o', '--output_path', type=str, default='./yolo_format', help='the file output path of LabelMe json format to yolov5 format!', metavar='lableme_json_output_path')
    parser.add_argument(
        "-c",
        "--count_output_file",
        type=str,
        default="./Statistics.txt",
        help="The output name to save quantity statistics infomation",
        metavar="CA_count_output_path",
    )

    opt = parser.parse_args()

    labelme2yolo(opt.input_path, opt.output_path, opt.count_output_file)
