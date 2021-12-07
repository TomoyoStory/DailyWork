# 针对YOLO的Anchor Based的架构进行聚类

import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def cluster_yolo2anchor(label_path, width, height, output_anchor_file='./yolo_anchor.txt', cluster_number=9, cluster_init='k-means++', n_init=5): # n_init指定Kmeans运行的次数
    '''
    对整体框进行聚类，获取最佳框的输出

    Args:
        label_path: 保存的yolo格式的txt文件夹的位置
        width: 原始标签对应图像的宽度
        height: 原始标签对应的图像的高度
        output_anchor_file: 输出聚类anchor的文件名绝对路径
        cluster_number: 聚类的anchor个数
        cluster_init: 初始化KMeans所使用的方法
        n_init: 聚类的次数

    Returns:
        None
    '''
    label_path = Path(label_path)
    output_anchor_file = Path(output_anchor_file)

    bbox = []
    for file in tqdm(list(label_path.iterdir()), desc=f'Reading file from {str(label_path)}', unit='files'):
        if file.is_file() and file.suffix == '.txt':
            with open(file, 'r') as f:
                for oneline in f.readlines():
                    label = oneline.strip().split()
                    bbox.append([x for x in label[3:]]) # 添加框
    bbox = np.array(bbox, dtype=np.float32)
    weight = np.array([width, height], dtype=np.float32)
    bbox = bbox * weight
    kmeans = KMeans(init=cluster_init, n_clusters=cluster_number, n_init=n_init, random_state=0, max_iter=1000) # 最大KMeans迭代次数
    logging.info('Clustering...')
    kmeans.fit(bbox)
    centroids = kmeans.cluster_centers_
    centroids.sort(axis=0)
    centroids = np.round(centroids)
    print('Clustering centers are:')
    with open(output_anchor_file,'w', encoding='utf-8') as f:
        for centroid in centroids:
            print(centroid)
            f.write(str(centroid) + '\n')
    logging.info('Cluster have been done! Good Luck!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Kmeans to cluster the Yolo format files to get Anchors", epilog="Anchor based style~ Emmmmmm...")
    parser.add_argument('-p', '--label_path', type=str, required=True, help='Yolo dataset label path', metavar='yolo_lable_path')
    parser.add_argument('-o', '--output_anchor_file', type=str, default='./yolo_anchor.txt', help='output file name', metavar='output_anchor_name')
    parser.add_argument('-wi', '--width', type=int, required=True, help='image width.', metavar='image_width')
    parser.add_argument('-he', '--height', type=int, required=True, help='image height.', metavar='image_height')
    # parser.add_argument('-p', '--label_path',  type=str, default='./', help='Yolo dataset label path', metavar='yolo_lable_path') # 便于在IDE中运行
    # parser.add_argument('-wi', '--width', type=int, default=800, help='image width', metavar='image_width')
    # parser.add_argument('-he', '--height', type=int, default=600, help='image height', metavar='image_height')
    parser.add_argument('-n', '--number',  type=int, default=9, help='number of anchors!', metavar='yolo_anchors_number')
    parser.add_argument('-i', '--init', type=str, default='k-means++', help='Kmeans init, "random" or "k-means++"(default)', metavar='kmeans_init')
    opt = parser.parse_args()

    cluster_yolo2anchor(opt.label_path, opt.width, opt.height, opt.output_anchor_file, opt.number, opt.init)