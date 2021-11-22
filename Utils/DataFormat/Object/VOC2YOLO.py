# VOC的xml格式转换为YOLO的xywh格式(2021-11-19 11:25:23 修正)

import os
import argparse
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ["persoon","car"] #! 根据数据标注情况选取实际的需要的类别

def xyxy2xywh(size, box):
    '''
    将xyxy格式数据转换为xywh格式

    Args:
        size: W H,图像宽高
        box: xyxy格式数据
    
    Returns:
        xywh格式数据
    '''
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)


def voc2yolo(input_path, output_path, width=None, height=None, save_difficult=False):
    '''
    将VOC的xml格式数据转换为YOLOv5的xywh格式

    Args:
        input_path: 输入xml标签的路径
        output_path: 输出txt标签的路径
        width: 自行设置的覆盖内部XML宽高的宽
        height: 自行设置的覆盖内部XML宽高的高
        save_difficult: 是否保留难样例
    
    Returns:
        None
    '''
    for file in tqdm(os.listdir(input_path), desc='Changing VOC format to YOLO format!'):
        label_file = input_path + os.sep + file
        if os.path.isfile(label_file) and Path(label_file).suffix.lower()[1:] == 'xml': # 指定仅读取xml
            with open(output_path + os.sep + file.replace('xml', 'txt'), 'w') as out_file:
                tree = ET.parse(label_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text) #! 同学给的原始VOC中宽高存在为零的问题,通过自行设置width和height覆盖解决这个问题
                h = int(size.find('height').text)
                if width:
                    w = width
                if height:
                    h = height

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if save_difficult: # 是否过滤难样例
                        if cls not in CLASSES:
                            continue
                    else:
                        if cls not in CLASSES or int(difficult) == 1:
                            continue
                    cls_id = CLASSES.index(cls)
                    bndbox = obj.find('bndbox')
                    box = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text),
                        float(bndbox.find('ymax').text)]
                    bbox = xyxy2xywh((w, h), box)
                    out_file.write(str(cls_id) + " " + " ".join('%.6f'%x for x in bbox) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change VOC dataset format to yolo dataset format! Please Change the CLASSES in PYTHON FILE for your class!", epilog="Hello! AI dataset! Hummmm....")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='the file input path of VOC format to yolov5 format!', metavar='voc_input_path')
    parser.add_argument('-o', '--output_path', type=str, default='./yolo_format', help='the file output path of VOC format to yolov5 format!', metavar='voc_output_path')  
    parser.add_argument('-W', '--width', type=int, help='the image width, it will cover the width in xml file!', metavar='cover_width')
    parser.add_argument('-H', '--height', type=int, help='the image height, it will cover the height in xml file!',  metavar='cover_height')
    # 便于在IDE中运行
    # parser.add_argument('-i', '--input_path', type=str, default='./', help='the file input path of VOC format to yolov5 format!', metavar='voc_input_path')
    # parser.add_argument('-o', '--output_path', type=str, default='./', help='the file output path of VOC format to yolov5 format!', metavar='voc_output_path')
    # parser.add_argument('-W', '--width', type=int, default=800, help='the image width, it will cover the width in xml file!', metavar='cover_width')
    # parser.add_argument('-H', '--height', type=int, default=600, help='the image height, it will cover the height in xml file!',  metavar='cover_height')
    
    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)
    voc2yolo(opt.input_path, opt.output_path, opt.width, opt.height)
