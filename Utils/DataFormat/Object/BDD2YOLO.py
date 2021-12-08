# BDD数据集格式转换为YOLO的xywh格式

import os
import json
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def bdd_to_yolo(bdd_label_path: str, 
                yolo_label_path: str,
                categorys: list,
                width: int=1280,
                height: int=720) -> None:
    '''
    将BDD数据集格式转换为YOLO格式，注意，这里的格式输出为x_center,y_center,w,h

    Args:
        bdd_label_path: bdd标签位置，该目录下应该有bdd100k_labels_images_train.json与bdd100k_labels_images_val.json文件
        yolo_label_path: 输出yolo格式的标签目录
        categorys: 包含的数据类别
        width: 图像的宽度
        height: 图像的高度
    
    Returns:
        None
    '''
    label_path = bdd_label_path + os.sep + "bdd100k_labels_images_%s.json"
    categorys_dict = {}  # 类别映射字典
    for i, category in enumerate(categorys):
        categorys_dict.update({category:i})

    for trainval in ['val','train']:  
        json_path = label_path % trainval
        yolo_path = os.path.join(yolo_label_path, trainval)
        os.makedirs(yolo_path,exist_ok=True)
        logging.info('Reading %s json file'%trainval)
        with open(json_path) as f:
            j = f.read()
        data = json.loads(j)
        
        for datum in tqdm(data, desc='Writing %s yolo format file'%trainval, unit='files'):
            file_str = '' # 保存每个文件所需的字段
            for label in datum['labels']:
                box2d = label.get('box2d') # 存在box2d
                if box2d:
                    label = bdd100k_labels_process(label) #! 这里会根据需求改变
                    x1 = float(round(box2d['x1']) / width)
                    y1 = float(round(box2d['y1']) / height)
                    x2 = float(round(box2d['x2']) / width)
                    y2 = float(round(box2d['y2']) / height)
                    x_center = (x1 + x2)/2
                    y_center = (y1 + y2)/2
                    w = max(x1,x2) - min(x1,x2)
                    h = max(y1,y2) - min(y1,y2)
                    # 默认转变的位数是6位
                    file_str = file_str + str(categorys_dict[label]) + ' '+ ' '.join(("%.6f" % x_center,"%.6f" % y_center,"%.6f" % w,"%.6f" % h)) + '\n'
            yolo_filename = os.path.splitext(datum['name'])[0] + '.txt'
            with open(os.path.join(yolo_path, yolo_filename), 'w') as f:
                f.write(file_str)
    logging.info('All Finish! ~~~///(^v^)\\\~~~ ,233~')


def get_bdd_categorys(bdd_label_path: str, output_path: str) -> None:
    '''
    获取BDD数据集的整体类别与对应类别数量，并根据str名称进行排序输出具体类别和对应个数

    Args:
        bdd_label_path: BDD标签路径(该目录下应该有bdd100k_labels_images_train.json与bdd100k_labels_images_val.json文件)
        output_path: output输出路径
    
    Returns:
        categorys(list):排序后的类别
    '''
    label_path = bdd_label_path + os.sep + "bdd100k_labels_images_%s.json"
    all_categorys = [] # 训练集验证集所有类别
    for trainval in ['val','train']:
        categorys = {}
        json_path = label_path % trainval

        with open(json_path) as f:
            j = f.read()
        data = json.loads(j)

        for datum in tqdm(data, desc='Counting %s dataset categories and number'%trainval): # 每张图
            for label in datum['labels']:
                if label.get('box2d'): # 存在box2d
                    label = bdd100k_labels_process(label) #! 这里会根据需求改变
                    if label not in categorys.keys():
                        categorys[label] = 0
                    categorys[label] += 1
                    if label not in all_categorys:
                        all_categorys.append(label)
                    
        with open(output_path + os.sep + trainval + '_categorys_count.txt','w') as f:
            keys = list(categorys.keys())
            keys.sort()
            for key in keys:
                f.write(key + " : " + str(categorys[key]) + "\n")

    logging.info('writing bdd100k.names file to %s' % os.path.join(output_path, 'bdd100k.names'))
    with open(os.path.join(output_path, 'bdd100k.names'),'w') as f:
        for category in all_categorys:
            f.write(category+'\n')
    return all_categorys


def bdd100k_labels_process(labels: dict) -> str:
    '''
    处理原始的字典中的相关参数。 #! 该函数根据需求可进行输出的各类变动

    Args:
        labels：labels代表原始json中的labels数组

    Returns:
        labels['category']:原始标签处理后的类别名称
    
    '''
    if labels['category'] == "traffic light": 
        #交通灯的颜色
        color = labels['attributes']["trafficLightColor"]
        if color == "none":
            labels['category'] = 'traffic light none'
        if color == "green":
            labels['category'] = 'traffic light green'
        if color == "yellow":
            labels['category'] = 'traffic light yellow'
        if color == "red":
            labels['category'] = 'traffic light red'
    
    # rider和person现在不做区分
    if labels['category'] == "rider":
        labels['category'] = "person"
    
    return labels['category']


def get_bdd_categorys_from_file(bdd100k_file_path: str) -> list:
    '''
    从名称文件中获取BDD数据集的类别

    Args:
        bdd100k_file_path：保存的.names名称文件路径

    Returns:
        categorys(list): 类别list
    '''
    categorys = []
    with open(bdd100k_file_path,'r') as f:
        for line in f.readlines():
            if len(line):
                categorys.append(line.strip())
    logging.info('get the whole categorys from %s'%bdd100k_file_path)
    return categorys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change bdd100k dataset format to yolo dataset format!", epilog="Hello! AI dataset! Hummmm....")
    parser.add_argument('-p', '--label_path', type=str, required=True, help='BDD dataset label path', metavar='bdd_lable_path')
    # parser.add_argument('-p', '--label_path',  type=str, default=r'./',help='BDD dataset label path', metavar='bdd_lable_path') # 便于在IDE中运行
    parser.add_argument('-ol', '--output_label_path', type=str, default='./yolo_format', help='YOLO format label output path', metavar='output_label_path')
    parser.add_argument('-oc', '--output_class_path', type=str, default='./', help='YOLO format label class output path', metavar='output_class_path')
    parser.add_argument('-n', '--names_file', type=str, default='./bdd100k.names', help='BDD100K names file', metavar='bdd100k.names')
    opt = parser.parse_args()

    if not os.path.isdir(opt.label_path):
        raise Exception("label_path is not a dir!")
    if not os.path.exists(opt.output_label_path):
        os.mkdir(opt.output_label_path)
    if not os.path.exists(opt.names_file):
        logging.info('%s path file don not exist:' % opt.names_file)
        categorys = get_bdd_categorys(opt.label_path, opt.output_class_path)
    else:
        categorys = get_bdd_categorys_from_file(opt.names_file)
        
    bdd_to_yolo(opt.label_path, opt.output_label_path, categorys)