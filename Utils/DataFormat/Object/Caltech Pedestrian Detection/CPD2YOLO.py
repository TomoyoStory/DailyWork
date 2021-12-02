# 将Caltech Pedestrian Detection数据集的VBB标注文件转换为YOLOv5格式

import glob
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def _vbb_anno2dict(vbb_file, images_output_path, image_input_path, filter_area=600):
    '''
    读取目录下单独的.vbb文件，处理获取该.vbb文件对应的的.seq图像序列中的数据,并过滤比较小的面积目标

    Args:
        vbb_file: 输入文件绝对路径，该文件为后缀的.vbb的标签文件
        images_output_path: 输入图像路径，该路径为通过CPD2Image.py脚本处理后得到的图像数据路径，该路径便于根据处理情况筛选有标签的图像
        image_input_path: 输出路径，该路径下包括标准的YOLOv5格式的images和labels文件夹
        filter_area: 过滤标签的面积，从而避免较小的标签
    
    Returns:
        annos:得到输出标签所对应的字典，具体字典内容参考源码
    '''
    #! 人太小了根本就看不清，设定就是600个像素单位作为过滤
    annos = {}
    vbb = loadmat(vbb_file) # Matlab Mat
    images_output_path.mkdir(exist_ok=True, parents=True)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0] # 得到每一帧图的目标检测类别
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]  # 可查看所有类别
    # person index
    person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为'person'的类别
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0: # 序列图中可能没有人的目标
            frame_name = images_output_path.joinpath(vbb_file.parent.name + '_' + vbb_file.stem + '_CPD_%07d.jpg'%frame_id)
            frame_dict = { frame_name:{
                                        'label':'person',
                                        'occlusion':[], # 0为未遮挡，1为遮挡 #^ 当前标签还未使用
                                        'bbox':[],
                                        'src_img_path':image_input_path.joinpath(vbb_file.parent.name).joinpath(vbb_file.stem).joinpath('CPD_%07d.jpg'%frame_id)
                                      } }
            for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                id = int(id[0][0]) - 1  # Maltab索引从1开始，而不是零
                if not id in person_index_list:  # 仅使用person标签的类别
                    continue
                pos = pos[0].tolist()
                area = pos[2] * pos[3] 
                if area < filter_area: # filter
                    continue
                occl = int(occl[0][0])
                frame_dict[frame_name]["occlusion"].append(occl)
                frame_dict[frame_name]["bbox"].append(pos)

            if frame_dict[frame_name]["bbox"]:
                annos.update(frame_dict)
    return annos


def vbb2yolo(input_path, image_input_path, output_path, filter_area):
    '''
    读取目录下所有的.vbb文件，将根据转换得到的数据信息，进行图像的复制和标签的获取

    Args:
        input_path: 输入路径，该路径下包括set00到set10的所有数据集的解压标签数据，下面包含.vbb数据
        image_input_path: 输入图像路径，该路径为通过CPD2Image.py脚本处理后得到的图像数据路径，该路径便于根据处理情况筛选有标签的图像
        output_path: 输出路径，该路径下包括标准的YOLOv5格式的images和labels文件夹
        filter_area: 过滤标签的面积，从而避免较小的标签
    
    Returns:
        None
    '''
    input_path = Path(input_path)
    assert input_path.is_dir(), f'{str(input_path)} is not a path!'
    image_input_path = Path(image_input_path)
    output_path = Path(output_path)
    images_output_path = output_path.joinpath('images')
    labels_output_path = output_path.joinpath('labels')
    labels_output_path.mkdir(exist_ok=True, parents=True)

    logging.info('Searching the .vbb file to get the labels')
    vbb_files = glob.glob(str(input_path.joinpath('**')), recursive=True)
    vbb_files = [Path(x) for x in vbb_files if x.split('.')[-1].lower()=='vbb'] # 获取seq文件图像路径

    # TODO 多进程加速数据获取
    for vbb_file in tqdm(vbb_files, desc='Geting the YOLOv5 format images and labels'):
        annos = _vbb_anno2dict(vbb_file, images_output_path, image_input_path, filter_area)
        if annos:
            for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                label_str = ''
                for bbox in anno['bbox']:
                    x_center = (bbox[0] + bbox[2]/2) / 640  # 数据集像素宽度640
                    y_center = (bbox[1] + bbox[3]/2) / 480  # 数据集像素高度480
                    width = bbox[2] / 640
                    height = bbox[3] / 480
                    label_str = label_str + '0 '+ ' '.join(("%.6f" % x_center,"%.6f" % y_center,"%.6f" % width,"%.6f" % height)) + '\n' #! 这里的0仅代表一类，即行人
                with open(labels_output_path.joinpath(filename.stem + '.txt'),'w', encoding='utf-8') as f:
                    f.write(label_str)
                shutil.copy(anno['src_img_path'], filename)

    logging.info('All Finish! (*╹▽╹*),HaHa~ There are so many images that are useless! You should just get part of it!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change Caltech Pedestrian Detection Dataset annotations file to images!", epilog="It is a little complex!")
    # parser.add_argument('-i', '--input_path', type=str, required=True, help='Caltech Pedestrian Detection Dataset annotations file', metavar='CPD_input_path')
    parser.add_argument('-i', '--input_path', type=str, default='./CPD_annotations_PATH', help='Caltech Pedestrian Detection Dataset annotations file', metavar='CPD_input_path') #  For IDE
    parser.add_argument('-im', '--image_input_path', type=str, default='/CPD_images', help='Caltech Pedestrian Detection Dataset images dir', metavar='CPD_image_dir')
    parser.add_argument('-o', '--output_path', type=str, default='./CPD_DATASET', help='Caltech Pedestrian Detection Dataset output path to get annotations', metavar='output_annotations_path')
    parser.add_argument('-f', '--filter_area', type=int, default=600, help='Caltech Pedestrian Detection Dataset filter area to get big enough label', metavar='label_filter_area')
    opt = parser.parse_args()
    
    vbb2yolo(opt.input_path, opt.image_input_path, opt.output_path, opt.filter_area)