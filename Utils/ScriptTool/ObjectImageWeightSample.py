# 根据标签进行图像权重采样，并一部分作为训练集，一部分作为验证集

import random
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def object_image_weight_sample(imgs_input_path, labels_input_path, imgs_output_path, labels_output_path, sample_scale=0.04, train_sacle=0.75):
    '''
    根据图像对应标签个数进行采样，采样比例

    Args:
        imgs_input_path: 输入图片的路径
        labels_input_path: 输入图片对应的标签的路径
        imgs_output_path: 输出图片的路径
        labels_output_path: 输入图片对应的标签的路径
        sample_scale: 采样的比例
        train_sacle: 训练集最后占整个的比例
    
    Returns:
        None
    '''
    imgs_input_path = Path(imgs_input_path)
    labels_input_path = Path(labels_input_path)
    imgs_output_path = Path(imgs_output_path)
    labels_output_path = Path(labels_output_path)
    imgs_output_path.joinpath('train').joinpath('images').mkdir(exist_ok=True, parents=True)
    imgs_output_path.joinpath('val').joinpath('images').mkdir(exist_ok=True, parents=True)
    labels_output_path.mkdir(exist_ok=True, parents=True)
    labels_output_path.joinpath('train').joinpath('labels').mkdir(exist_ok=True, parents=True)
    labels_output_path.joinpath('val').joinpath('labels').mkdir(exist_ok=True, parents=True)

    logging.info('Getting the files path')
    labels_list = list(labels_input_path.glob('*.txt'))
    
    list_weight = []
    for x in tqdm(labels_list, desc='Getting the lables number of each file!'): # TODO 多进程加速运算速度
        with open(x, '', encoding='utf-8') as f:
            file_str = f.read()
            count = len(file_str.split('\n')) - 1
            list_weight.append(count)
    
    indices = random.choices(range(len(labels_list)), weights=list_weight, k=int(sample_scale*len(labels_list)))  # 根据权重随机采样
    train_indices = indices[:int(0.75*len(indices))]
    val_indices = indices[int(0.75*len(indices)):]

    # 训练集复制
    for x in tqdm(train_indices, desc='Copying Train Dataset'):
        src_label = labels_list[x]
        dst_label = labels_output_path.joinpath('train').joinpath('labels').joinpath(src_label.name)
        src_image = imgs_input_path.joinpath(src_label.stem + '.jpg') #^ 图像格式后缀根据情况修改
        dst_image = imgs_output_path.joinpath('train').joinpath('images').joinpath(src_label.stem + '.jpg')
        shutil.copy(src_label, dst_label)
        shutil.copy(src_image, dst_image)
    
    # 验证集复制
    for x in tqdm(val_indices, desc='Copying Val Dataset'):
        src_label = labels_list[x]
        dst_label = labels_output_path.joinpath('val').joinpath('labels').joinpath(src_label.name)
        src_image = imgs_input_path.joinpath(src_label.stem + '.jpg') #^ 图像格式后缀根据情况修改
        dst_image = imgs_output_path.joinpath('val').joinpath('images').joinpath(src_label.stem + '.jpg')
        shutil.copy(src_label, dst_label)
        shutil.copy(src_image, dst_image)
    
    logging.info('All Finish! (*╹▽╹*),HaHa~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample images to train dataset and val dataset depending on bbox number!", epilog="For the sample balance!")
    # parser.add_argument('-i', '--input_path', type=str, required=True, help='the path of source image', metavar='src_input_path')
    # parser.add_argument('-l', '--labels_input_path', type=str, default='./', help='the inputpath of labels!', metavar='labels_input_path')
    # parser.add_argument('-i', '--imgs_input_path', type=str, default='./', help='the inputpath of source image', metavar='imgs_input_path') # For IDE
    # parser.add_argument('-l', '--labels_input_path', type=str, default='./', help='the inputpath of labels!', metavar='labels_input_path')
    parser.add_argument('-io', '--imgs_output_path', type=str, default='./', help='the outpath of source image', metavar='imgs_output_path')
    parser.add_argument('-lo', '--labels_output_path', type=str, default='./', help='the outpath of labels!', metavar='labels_output_path')
    parser.add_argument('-s', '--sample_scale', type=float, default=0.04, help='the scale of sample!', metavar='sample_scale')
    parser.add_argument('-ts', '--train_sacle', type=float, default=0.75, help='the scale of train dataset in all data!', metavar='train_sacle')
    opt = parser.parse_args()

    object_image_weight_sample(opt.imgs_input_path, opt.labels_input_path, opt.imgs_output_path, opt.labels_output_path, opt.sample_scale, opt.train_sacle)