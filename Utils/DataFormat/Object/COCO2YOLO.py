# COCO数据集转换为YOLO的xywh格式

import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# COCO_CLASS = ['person','bicycle','car','motorcycle','airplane','bus',
#               'train','truck','boat','traffic light','fire hydrant',
#               'stop sign','parking meter','bench','bird','cat','dog',
#               'horse','sheep','cow','elephant','bear','zebra','giraffe',
#               'backpack','umbrella','handbag','tie','suitcase','frisbee',
#               'skis','snowboard','sports ball','kite','baseball bat',
#               'baseball glove','skateboard','surfboard','tennis racket',
#               'bottle','wine glass','cup','fork','knife','spoon','bowl',
#               'banana','apple','sandwich','orange','broccoli','carrot','hot dog',
#               'pizza','donut','cake','chair','couch','potted plant',
#               'bed','dining table','toilet','tv','laptop','mouse','remote',
#               'keyboard','cell phone','microwave','oven','toaster','sink',
#               'refrigerator','book','clock','vase','scissors','teddy bear',
#               'hair drier','toothbrush'] #^ 需要提取的coco数据集中的类
COCO_CLASS = ["traffic light"]  # ^ 仅采样交通灯


def coco_to_yolo(coco_image_path: str, coco_label_path: str, output_path: str) -> None:
    """
    将COCO的json格式数据转换为YOLOv5的xywh格式

    Args:
        coco_image_path: COCO数据集对应的图像所在的位置(训练接和验证集图像应该和标签相互对应)
        coco_label_path: COCO数据集对应的标签所在的位置(训练接和验证集图像应该和标签相互对应)
        output_path: 输出图像和标签的位置

    Returns:
        None
    """
    coco_image_path = Path(coco_image_path)
    coco_label_path = Path(coco_label_path)
    output_path = Path(output_path)
    image_output_path = output_path.joinpath("images")
    label_output_path = output_path.joinpath("labels")
    image_output_path.mkdir(exist_ok=True, parents=True)
    label_output_path.mkdir(exist_ok=True, parents=True)

    logging.info("Initializing COCO Class from %s" % str(coco_label_path))
    coco = COCO(coco_label_path)  # 初始化COCO类
    categorys_dict = {}
    for i, category in enumerate(COCO_CLASS):
        categorys_dict.update({category: i})

    category_number = {}  # ^ 用于记录各个类别的个数
    for category in tqdm(
        COCO_CLASS, desc="Coping images and Writing labels", unit="category"
    ):
        # logging.info('Writing Class ---%s--- to yolo format' % category)
        category_id = coco.getCatIds(category)
        category_number.update(
            {category: str(len(coco.getAnnIds(catIds=category_id)))}
        )  # ^ 类别个数统计
        img_ids = coco.getImgIds(catIds=category_id)
        imgs = coco.loadImgs(img_ids)
        for img in imgs:
            file_str = ""  # ^ 保存当前类别字段
            annIds = coco.getAnnIds(
                imgIds=img["id"], catIds=category_id, iscrowd=None
            )  # ^ iscrowd属性根据实际情况自行选择
            anns = coco.loadAnns(annIds)
            for ann in anns:
                x_center = (ann["bbox"][0] + ann["bbox"][2] / 2) / img["width"]
                y_center = (ann["bbox"][1] + ann["bbox"][3] / 2) / img["height"]
                w = ann["bbox"][2] / img["width"]
                h = ann["bbox"][3] / img["height"]
                file_str = (
                    file_str
                    + str(categorys_dict[category])
                    + " "
                    + " ".join(
                        ("%.6f" % x_center, "%.6f" % y_center, "%.6f" % w, "%.6f" % h)
                    )
                    + "\n"
                )
            yolo_output_label_name = label_output_path.joinpath(
                Path(img["file_name"]).stem + ".txt"
            )
            # 追加写标签
            with open(yolo_output_label_name, "a") as f:
                f.write(file_str)
            # 复制图片
            input_image_path = Path(coco_image_path).joinpath(Path(img["file_name"]))
            ouput_image_path = Path(image_output_path).joinpath(Path(img["file_name"]))
            if not ouput_image_path.exists():
                shutil.copy(input_image_path, ouput_image_path)

    logging.info("Writing category number to file")
    with open(output_path.joinpath("category_number.txt"), "w") as f:
        keys = list(category_number.keys())
        keys.sort()
        for key in keys:
            f.write(key + " : " + str(category_number[key]) + "\n")

    logging.info("Writing category name and id to file")  # ^ 写类别和对应的id信息
    with open(output_path.joinpath("category.names"), "w") as f:
        for i, category in enumerate(COCO_CLASS):
            f.write(category + " : " + str(i) + "\n")

    logging.info("All COCO dataset category are finished! ~~~///(^v^)\\\~~~ ,haha~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change COCO dataset format to yolo dataset format! Please select the CLASSES in PYTHON FILE for your class!",
        epilog="Hello! AI dataset! Hummmm....",
    )
    # parser.add_argument('-i', '--image_path', type=str, required=True, help='the image path of COCO dataset!', metavar='coco_image_path')
    # parser.add_argument('-l', '--label_path', type=str, required=True, help='the label path of COCO dataset!', metavar='coco_label_path')
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="./train2017",
        help="the image path of COCO dataset!",
        metavar="coco_image_path",
    )  # FOR IDE
    parser.add_argument(
        "-l",
        "--label_path",
        type=str,
        default="./annotations_trainval2017/annotations/instances_train2017.json",
        help="the label path of COCO dataset!",
        metavar="coco_label_path",
    )  # FOR IDE
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./data/train",
        help="the file output path of COCO format to yolov5 format!",
        metavar="coco_output_path",
    )
    opt = parser.parse_args()

    coco_to_yolo(opt.image_path, opt.label_path, opt.output_path)
