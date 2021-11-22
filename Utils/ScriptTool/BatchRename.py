# 批量修改目录下的名称，以起始索引开始进行名称修改

import os
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change all the names of the files in dir to fix format!", epilog="It is easy! O(∩_∩)O~")
    parser.add_argument('-p', '--path', type=str, default='./', required=True, help='the file path to change name!', metavar='file_path')
    # parser.add_argument('-p', '--path', type=str, default='./', help='the file path to change name!', metavar='file_path') # For IDE
    parser.add_argument('-s', '--start_pos', type=int, default=0, help='the start index to increase', metavar='start_index')
    opt = parser.parse_args()

    start_pos = opt.start_pos # start index
    for file in tqdm(os.listdir(opt.path), desc='Changing the name'):
        current_file_name = os.path.join(opt.path, file)
        if os.path.isfile(current_file_name):
            new_name = os.path.join(opt.path, '%07d.jpg'%start_pos) #! 这里可以修改整个索引的起始位置
            os.rename(current_file_name, new_name)
            start_pos= start_pos + 1
    logging.info('All Finish! (*╹▽╹*),HaHa~')

