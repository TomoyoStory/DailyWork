# Caltech Pedestrian Detection数据集数据转换为可见图片，而不是.seq文件

import glob
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def seqs2images(input_path, output_path):
    '''
    读取目录下所有的.seq文件，将其转换为实际的图片

    Args:
        input_path: 输入路径，该路径下包括set00到set10的所有数据集的解压数据，下面包含.set数据
        output_path: 输出图片的路径
    
    Returns:
        None
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 文件递归搜寻
    logging.info('Searching the .seq file to get the images')
    files = glob.glob(str(input_path.joinpath('**')), recursive=True)
    files = [Path(x) for x in files if x.split('.')[-1].lower()=='seq'] # 获取seq文件图像路径

    for x in tqdm(files, desc='Geting the images!', unit='imgs'):
        parent_str = x.parent.name
        parent_dir = output_path.joinpath(parent_str).joinpath(x.stem)
        parent_dir.mkdir(exist_ok=True, parents=True)

        with open(x, 'rb') as f:
            string = str(f.read().decode('latin-1')) # 这里由于编码格式的问题，必须使用外国的latin-1解码
            splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
            strlist=string.split(splitstring)
        count = -1
        # 处理文件分段，除了第一个分段，每一个分段都是一张图
        # deal with file segment, every segment is an image except the first one
        for img in strlist:
            file_path = parent_dir.joinpath('CPD_%07d.jpg'%count) #^ 可修改输出的名称补零长度
            # 抛弃第一个代表的.seq文件的头
            # abandon the first one, which is filled with .seq header
            if count != -1:
                with open(file_path,'wb+') as f:
                    f.write(splitstring.encode('latin-1')) # 同上编码
                    f.write(img.encode('latin-1')) # 同上编码
            count += 1

    logging.info('All Finish! (*╹▽╹*),HaHa~')
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Change Caltech Pedestrian Detection Dataset .seq file to images!", epilog="It is a little complex!")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Caltech Pedestrian Detection Dataset set file', metavar='CPD_input_path')
    # parser.add_argument('-i', '--input_path', type=str, default='./CPD_PATH', help='Caltech Pedestrian Detection Dataset  set file', metavar='CPD_input_path') #  For IDE
    parser.add_argument('-o', '--output_path', type=str, default='./CPD_images', help='Caltech Pedestrian Detection Dataset output path to get images', metavar='output_images_path')
    opt = parser.parse_args()

    seqs2images(opt.input_path, opt.output_path)