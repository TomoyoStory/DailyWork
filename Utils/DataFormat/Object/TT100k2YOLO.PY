# TTK数据集格式转换为YOLO的xywh格式
# 这个关于TTK的脚本并没有对具体的标示牌类别进行区分，因为没有意义
# crop_multiclass选项仅仅用于裁剪中间的所有交通标示牌，并且根据文件名分类
# single_index则用于单纯的目标检测，索引值可根据自己与其他数据集的兼容性进行设置

import os
import json
import logging
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import Pool

NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def tt100k_to_yolo_single(annotations_path, output_label_path, single_index, width=2048, height=2048):
    '''
    该函数只提取标示牌，并不对标示牌的具体类别进行分类拆分,注意，这里的格式输出为x_center,y_center,w,h

    Args:
        annotations_path: annotations.json文件的路径
        output_label_path: yolo格式输出的路径
        single_index: ttk100k数据集并入其他数据集时候的索引
        width:图像的宽度
        height:图像的高度
    
    Returns:
        None
    '''
    logging.info('Reading %s' %annotations_path)
    with open(annotations_path) as f:
            j = f.read()
    data = json.loads(j)
    
    train_yolo_path = os.path.join(output_label_path, 'train')
    os.makedirs(train_yolo_path, exist_ok=True)
    test_yolo_path = os.path.join(output_label_path, 'test')    
    os.makedirs(test_yolo_path, exist_ok=True)
    
    for label in tqdm(data["imgs"],desc='Writing yolo format file, only for one class classed traffic sign!\n'):
        path = data["imgs"][label]['path']
        dataset = path.split('/')[0] # train, test, other
        # 训练集和验证集
        if dataset in ['train','test'] and len(data["imgs"][label]['objects']):
            file_str = ''
            for obj in data["imgs"][label]['objects']:
                x1 = obj['bbox']['xmin'] / width
                y1 = obj['bbox']['ymin'] / height
                x2 = obj['bbox']['xmax'] / width
                y2 = obj['bbox']['ymax'] / height
                x_center = (x1 + x2)/2
                y_center = (y1 + y2)/2
                w = x2 - x1
                h = y2 - y1
                file_str = file_str + str(single_index) + ' ' + ' '.join(("%.6f"%x_center, "%.6f"%y_center, "%.6f"%w, "%.6f"%h)) + '\n'
            yolo_filename = str(label) + '.txt'
            if dataset=='train':
                with open(os.path.join(train_yolo_path, yolo_filename), 'w') as f:
                    f.write(file_str)
            elif dataset=='test':
                with open(os.path.join(test_yolo_path, yolo_filename), 'w') as f:
                    f.write(file_str)

    logging.info('All Finish! (*￣︶￣) ,233~')


def tt100k_to_crop_multiclass(annotations_path, output_label_path):
    '''
    该函数提取标示牌，截取标示牌部分像素，并对具体的标示牌进行拆分分类,要求annotations.json目录下保存train和val目录(原始数据集保存格式)

    Args:
        annotations_path: annotations.json文件的路径
        output_label_path: yolov5格式输出的路径  

    Returns:
        None
    '''
    logging.info('Reading %s for multiclass' %annotations_path)
    with open(annotations_path) as f:
            j = f.read()
    data = json.loads(j)

    categorys = data['types']

    # 创建相关的目录
    output_label_path = os.path.abspath(output_label_path)
    image_multiclass_path = os.path.join(output_label_path, 'imgs')

    train_yolo_path = os.path.join(image_multiclass_path, 'train')
    os.makedirs(train_yolo_path,exist_ok=True) 
    val_yolo_path = os.path.join(image_multiclass_path, 'test')
    os.makedirs(val_yolo_path,exist_ok=True)
    for x in [train_yolo_path,val_yolo_path]:
        for traffic_name in categorys:
            os.makedirs(os.path.join(x, traffic_name),exist_ok=True) 
    
    train_categorys_count = _tt100K_categorys_count(categorys) # 初始化训练集整体类别计数字典
    test_categorys_count = _tt100K_categorys_count(categorys) # 初始化验证集整体类别计数字典

    info_list = [] # 保存所有的crop信息
    for label in tqdm(data["imgs"], desc='Getting the rectangle and ROI information', unit='batchs'): #^ 字典的for in语句得到的是key
        path = data["imgs"][label]['path']
        # 训练集与验证集
        dataset = path.split('/')[0] # train, test, other
        if dataset in ['train','test'] and len(data["imgs"][label]['objects']):
            absolute_path = os.path.join(os.path.dirname(annotations_path) , path)
            for obj in data["imgs"][label]['objects']:
                category = obj['category']
                x1 = obj['bbox']['xmin']
                y1 = obj['bbox']['ymin']
                x2 = obj['bbox']['xmax']
                y2 = obj['bbox']['ymax']
                box = [x1,y1,x2,y2]
                _ , extension= os.path.splitext(absolute_path)
                if dataset=='train':
                    region_path_name = os.path.join(train_yolo_path, str(category), category + '_' + str(train_categorys_count[category]).zfill(8) + extension)
                    train_categorys_count[category] += 1
                elif dataset=='test':
                    region_path_name = os.path.join(val_yolo_path, str(category), category + '_' + str(test_categorys_count[category]).zfill(8) + extension)
                    test_categorys_count[category] += 1
                info_list.append([absolute_path, box, region_path_name])

    _tt100K_categorys_count_output(os.path.join(output_label_path, 'train.count'), train_categorys_count)
    _tt100K_categorys_count_output(os.path.join(output_label_path, 'val.count'), test_categorys_count)

    desc='Getting the ROI and Save to corresponding dir!'
    with Pool(NUM_THREADS) as pool: # 多进程提升4倍速率
        pbar = tqdm(pool.imap(_crop_and_save, info_list), desc=desc, total=len(info_list))
        for i in pbar:
            pbar.update()
    logging.info('All Finish! (*╹▽╹*),HaHa~')
            

def _crop_and_save(info_item):
    absolute_path = info_item[0]
    box = info_item[1]
    region_path_name = info_item[2]
    with Image.open(absolute_path) as im:
        region = im.crop(box)
        region.save(region_path_name)


def _tt100K_categorys_count(categorys):
    '''
    统计每一类类别的数量，便于整体的命名

    Args:
        categorys: 整个类别
    
    Returns:
        categorys_count(dict): 每个类别的数量字典
    '''
    categorys_count = {}
    for category in categorys:
        categorys_count.update({category:0})
    return categorys_count


def _tt100K_categorys_count_output(names_path, categorys_count_dict):
    '''
    该函数用于类别计数，便于评估整体标注后相关标签的数量，便于数据分析与样本均衡

    Args:
        names_path: 保存文件的路径
        categorys_count_dict: 每个类别的数量字典
    
    Returns:
        None
    '''
    with open(names_path,'w') as f:
        for label in categorys_count_dict:
            oneline = label + ' : ' + str(categorys_count_dict[label]) + '\n'
            f.write(oneline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change TTK100k dataset format to yolo dataset format!", epilog="Hello AI dataset! Hummmm....")
    parser.add_argument('-p','--annotations_path', type=str, required=True, help='TT100k dataset annotations.json path', metavar='annotations_path')
    # parser.add_argument('-p', '--annotations_path', type=str, default='./annotations.json', help='TT100k dataset annotations.json path') # 为了方便IDE
    parser.add_argument('-o', '--output_label_path', type=str, default='./yolo_format_multiprocess', help='YOLO format label output path. Use multiprocess Now!', metavar='output_label_path')
    parser.add_argument('-s', '--single_index', type=int, default=0, help='TT100k single output class index (just for expanding dataset!)', metavar='class index for traffic sign')
    parser.add_argument('-c', '--crop_multiclass', action='store_true', help='only crop traffic sign for classfy or Not, TT100k output multiclass')
    # parser.add_argument('-c', '--crop_multiclass', action='store_true', default=True,  help='only crop traffic sign for classfy, TT100k output multiclass') # 为了方便IDE
    opt = parser.parse_args()

    if not os.path.exists(opt.annotations_path):
        raise Exception('%s file do not exists:' % opt.annotations_path)
    
    if opt.crop_multiclass:
        logging.info('tt100k dataset is classified to multiclass!')
        tt100k_to_crop_multiclass(opt.annotations_path, opt.output_label_path)
    else:
        logging.info('tt100k dataset is classified to single class with a index : %d' % opt.single_index )
        tt100k_to_yolo_single(opt.annotations_path, opt.output_label_path, opt.single_index)