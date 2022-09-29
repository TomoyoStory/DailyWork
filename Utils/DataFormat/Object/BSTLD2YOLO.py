# Bosch Small Traffic Lights Dataset 博世交通灯数据集转换为YOLO的xywh格式

import yaml
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def get_BSTLD_categorys(input_path: str,
                        output_path: str, 
                        BSTLD_labels_path: list = ['train.yaml', 'test.yaml','additional_train.yaml']) -> dict:
    '''
    统计Bosch Small Traffic Lights Dataset数据集的标签类别，并根据名称进行排序输出具体类别和对应个数

    Args:
        input_path: BSTLD数据集对应的根目录(该目录下包括additional_train.yaml,train.yaml,test.yaml和rgb图像目录)
        output_path: 输出图像和标签的位置
        BSTLD_labels_path: BSTLD数据集对应的所有标签文件
    
    Returns:
        keys: 输出统计类别名称和对应的ID
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    categorys_dict = {} #^ 类别获取和统计计数
    for label_file in BSTLD_labels_path:
        logging.info('Reading %s' % str(input_path.joinpath(label_file)))
        with open(input_path.joinpath(label_file), "r" , encoding="utf-8") as f:
            labels = yaml.load(f,Loader=yaml.FullLoader)
            logging.info('Reading %s is finished' % str(input_path.joinpath(label_file)))
            for label in tqdm(labels, desc='Counting %s dataset categories and number' % label_file, unit=' images'):
                for current_label in label['boxes']:
                    if current_label['label'] not in categorys_dict.keys():
                        categorys_dict.update({current_label['label']:1}) # 初始化类别
                    else:
                        categorys_dict[current_label['label']] += 1  # 类别数据加1
    
    logging.info('Writing Bosch Small Traffic Lights Dataset number of categories to %s' % str(output_path.joinpath('categorys_count.txt')))
    # 类别个数统计输出
    with open(output_path.joinpath('categorys_count.txt'),'w') as f:
        keys = list(categorys_dict.keys())
        keys.sort()
        for key in keys:
            f.write(key + " : " + str(categorys_dict[key]) + "\n")
    
    logging.info('writing BSTLD.names file to %s' % str(output_path.joinpath('BSTLD.names')))
    # 类别名称和ID输出
    keys_dict = {}
    with open(output_path.joinpath('BSTLD.names'),'w') as f:
        for i,key in enumerate(keys):
            f.write(key + " : " + str(i) + "\n")
            keys_dict.update({key:i}) #^ 输出类别名称和对应的ID
    
    return keys_dict

def BSTLD_to_yolo(input_path: str,
                  output_path: str,
                  keys_dict: dict,
                  width: int = 1280,
                  height: int = 720,
                  BSTLD_labels_path: list = ['train.yaml', 'test.yaml','additional_train.yaml']) -> None:
    '''
    将Bosch Small Traffic Lights Dataset的yaml格式数据转换为YOLOv5的xywh格式

    Args:
        input_path: BSTLD数据集对应的根目录(该目录下包括additional_train.yaml,train.yaml,test.yaml和rgb图像目录)
        output_path: 输出图像和标签的位置
        keys_dict: 标签名称和ID对应的定义字典
        width: 图像的宽度
        height: 图像的高度
        BSTLD_labels_path: BSTLD数据集对应的所有标签文件
    
    Returns:
        None
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    image_output_path = output_path.joinpath('images')
    label_output_path = output_path.joinpath('labels')
    image_output_path.mkdir(exist_ok=True, parents=True) #^ 构建路径
    label_output_path.mkdir(exist_ok=True, parents=True)

    for label_file in BSTLD_labels_path:

        # 构建每个训练集类别的文件目录
        current_file_image_output_path = image_output_path.joinpath(Path(label_file).stem)
        current_file_label_output_path = label_output_path.joinpath(Path(label_file).stem)
        current_file_image_output_path.mkdir(exist_ok=True, parents=True)
        current_file_label_output_path.mkdir(exist_ok=True, parents=True)

        logging.info('Reading %s' % str(input_path.joinpath(label_file)))
        with open(input_path.joinpath(label_file), "r" , encoding="utf-8") as f:
            labels = yaml.load(f, Loader = yaml.FullLoader)
            logging.info('Reading %s is finished' % str(input_path.joinpath(label_file)))
            for label in tqdm(labels, desc='Changing %s\'s BSTLD format to yolo format and Copying the images which have trffic light' % label_file, unit=' images'):
                label_str = '' #^ 标签对应字段
                for current_label in label['boxes']:
                    class_id = keys_dict[current_label['label']]
                    x_center = ((current_label['x_max'] + current_label['x_min']) / 2) / width
                    y_center = ((current_label['y_max'] + current_label['y_min']) / 2) / height
                    w = (current_label['x_max'] - current_label['x_min']) / width
                    h = (current_label['y_max'] - current_label['y_min']) / height
                    label_str = label_str + str(class_id) + ' '+ ' '.join(("%.6f" % x_center,"%.6f" % y_center,"%.6f" % w,"%.6f" % h)) + '\n'
                    
                if len(label['boxes']) != 0:
                    if label_file == 'test.yaml':
                        image_path = input_path.joinpath('rgb').joinpath('test').joinpath(Path(label['path']).name) #! 需要特别注意的是,test.yaml的路径格式和其他两个不一样,给的是url地址,所以这里需要根据实际的情况进行转换
                    else:
                        image_path = input_path.joinpath(label['path']) #^ 当前对应图像的绝对路径

                    with open(current_file_label_output_path.joinpath(image_path.stem + '.txt'), 'w', encoding='utf-8') as f:
                        f.write(label_str)                    
                    shutil.copy(image_path, current_file_image_output_path.joinpath(image_path.name)) #^ 复制图片
        
        logging.info('Process %s is finished' % str(input_path.joinpath(label_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change Bosch Small Traffic Lights Dataset format to yolo dataset format!", epilog="Hello! AI dataset! Hummmm....")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='the root dir of Bosch Small Traffic Lights Dataset!', metavar='BSTLD_root_dir')
    # parser.add_argument('-i', '--input_path', type=str, default='./Bosch_Small_Traffic_Lights_Dataset', help='the root dir of Bosch Small Traffic Lights Dataset!', metavar='BSTLD_root_dir')  # FOR IDE
    parser.add_argument('-l', '--label_path', type=str, required=True, help='the label of Bosch Small Traffic Lights Dataset! (relative path)', metavar='BSTLD_label_path')
    # parser.add_argument('-l', '--label_path', type=str, default='train.yaml', help='the label of Bosch Small Traffic Lights Dataset! (relative path)', metavar='BSTLD_label_path')  # FOR IDE
    parser.add_argument('-o', '--output_path', type=str, default='/data/Bosch_Small_Traffic_Lights', help='the file output path of Bosch Small Traffic Lights Dataset format to yolov5 format!', metavar='BSTLD_output_path')
    opt = parser.parse_args()

    keys_dict = get_BSTLD_categorys(opt.input_path, opt.output_path) #^ 统计标签信息
    BSTLD_to_yolo(opt.input_path, opt.output_path, keys_dict)