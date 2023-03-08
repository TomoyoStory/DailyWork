# 批量修改目录下的名称,以起始索引开始进行名称修改

import os
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def batch_rename(path: str, start_pos: int = 0, suffix: str = '*.jpg') -> None:
    '''
    批量文件命名格式修改,该函数完成针对path路径下的文件名称修改为标准0000001.jpg的数据名称(该函数需要根据需要进行适配修改)

    Args:
        path: 输入批量名称修改的路径,该路径下必须包含对应的数据
        start_pos: 批量命名的起始位置
        suffix: 对应的后缀名

    Returns:
        None
    '''
    path = Path(path)
    for file in tqdm(path.glob(suffix), desc='Changing the name', unit='files'):   
        if file.is_file():
            new_name = file.parent.joinpath('%07d.jpg'%start_pos) #! 这里可以修改整个索引的起始位置和后缀
            os.rename(file, new_name)
            start_pos= start_pos + 1
    logging.info('All Finish! (*╹▽╹*),HaHa~')

def batch_prefix_rename(path: str, suffix: str = '*.tfrecord', sep: str = '_') -> None:
    '''
    批量文件命名格式修改,该函数完成针对path路径下对应匹配模式的文件名的修改(该函数需要根据需要进行适配修改)

    Args:
        path: 输入批量名称修改的路径,该路径下必须包含对应的数据
        suffix: 对应的后缀名
        sep: 分割的字符

    Returns:
        None
    '''
    path = Path(path)
    if not path.is_dir():
        logging.warning('Current path string is not a dir, Please Check it!')
        return
    for file in tqdm(path.glob(suffix), desc='Changing the name', unit='files'):
        if file.is_file():
            name_split_list = file.name.split(sep = sep)
            name_split_list[0] = str(int(name_split_list[0]) - 33) #! 这里的模式根据需要自行修改
            new_name = sep.join(name_split_list)
            new_name = file.parent.joinpath(new_name)
            os.rename(file, new_name)
    logging.info('All Finish! (*╹▽╹*),HaHa~')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change all the names of the files in dir to fix format!", epilog="It is easy! O(∩_∩)O~")
    # parser.add_argument('-p', '--path', type=str, default='./', required=True, help='the file path to change name!', metavar='file_path')
    parser.add_argument('-p', '--path', type=str, default='/personal/polar_tfrecords/train', help='the file path to change name!', metavar='file_path') # For IDE 
    parser.add_argument('-s', '--start_pos', type=int, default=0, help='the start index to increase', metavar='start_index') # 按需使用
    parser.add_argument('-su','--suffix', type=str, default='*.tfrecord', help='find the file with a suffix matched!', metavar='suffix') # 按需显示使用
    opt = parser.parse_args()

    # batch_rename(opt.path, opt.start_pos)
    batch_prefix_rename(opt.path, opt.suffix) 