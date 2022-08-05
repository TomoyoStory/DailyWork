# CA目标检测数据集格式转换为YOLO数据集格式脚本

import glob
import json
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
# from multiprocessing.pool import Pool

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # 支持的图像后缀名
LABEL_FORMATS = ['txt']
# NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

#^ 以下定义根据CA标注需求和识别需求说明书定义,相关定义修改需严格遵照实际的需求和相关定义修改(常年根据实际情况改变)
# ID代表实例类别,TrainID代表参与训练的ID(-1代表不参与训练),TypeID表示类别ID

# 目标检测
OBJECT_DICT = {
    #       Name                         ID            TrainID             TypeID
    'vehicle_car-2dbox':{             'id':0,       'train_id':0,       'type_id':1   },  # 小汽车
    'vehicle_bus-2dbox':{             'id':1,       'train_id':1,       'type_id':1   },  # 大巴车
    'vehicle_truck-2dbox':{           'id':2,       'train_id':2,       'type_id':1   },  # 卡车
    'vehicle_tricycle-2dbox':{        'id':3,       'train_id':3,       'type_id':1   },  # 三轮车
    'vehicle_ridern-2dbox':{          'id':4,       'train_id':4,       'type_id':1   },  # 骑行车（无人）
    'vehicle_ridery-2dbox':{          'id':5,       'train_id':5,       'type_id':1   },  # 骑行车（有人）
    'vehicle_else-2dbox':{            'id':6,       'train_id':6,       'type_id':1   },  # 异形车辆
    'pedestrian_genral-2dbox':{       'id':7,       'train_id':7,       'type_id':2   },  # 普通行人
    'pedestrian_else-2dbox':{         'id':8,       'train_id':-1,      'type_id':2   },  # 异形人
    'animal_horse-2dbox':{            'id':9,       'train_id':-1,      'type_id':3   },  # 普通动物_马
    'animal_cattle-2dbox':{           'id':10,      'train_id':-1,      'type_id':3   },  # 普通动物_牛
    'animal_dog-2dbox':{              'id':11,      'train_id':-1,      'type_id':3   },  # 普通动物_狗
    'animal_else-2dbox':{             'id':12,      'train_id':-1,      'type_id':3   },  # 普通动物_其它
    'obstacle_cone-2dbox':{           'id':13,      'train_id':8,       'type_id':4   },  # 锥桶
    'obstacle_triangle-2dbox':{       'id':14,      'train_id':-1,      'type_id':4   },  # 紧急三角架
    'obstacle_waterhorse-2dbox':{     'id':15,      'train_id':9,       'type_id':4   },  # 水马
    'obstacle_column-2dbox':{         'id':16,      'train_id':10,      'type_id':4   },  # 警示柱子
    'obstacle_else-2dbox':{           'id':17,      'train_id':11,      'type_id':4   },  # 人工障碍物（其他）
    'marking_diversionline-2dbox':{   'id':18,      'train_id':12,      'type_id':5   },  # 导流带
    'marking_stop-2dbox':{            'id':19,      'train_id':13,      'type_id':5   },  # 停止线
    'marking_pavement-2dbox':{        'id':20,      'train_id':14,      'type_id':5   },  # 人行道
    'marking_speedbump-2dbox':{       'id':21,      'train_id':-1,      'type_id':5   },  # 减速带
    'guidearrow_genral-2dbox':{       'id':22,      'train_id':15,      'type_id':6   },  # 通用导向箭头
    'roadtext_genral-2dbox':{         'id':23,      'train_id':16,      'type_id':7   },  # 普通路面文字
    'roadsign_genral-2dbox':{         'id':24,      'train_id':17,      'type_id':8   },  # 交通标志牌_通用
    'roadsign_electron-2dbox':{       'id':25,      'train_id':18,      'type_id':8   },  # 交通标志牌_电子
    'roadsign_else-2dbox':{           'id':26,      'train_id':19,      'type_id':8   },  # 交通标志牌_其他
    'traffic_light_red-2dbox':{       'id':27,      'train_id':20,      'type_id':9   },  # 红灯
    'traffic_light_green-2dbox':{     'id':28,      'train_id':21,      'type_id':9   },  # 绿灯
    'traffic_light_yellow-2dbox':{    'id':19,      'train_id':22,      'type_id':9   },  # 黄灯
    'traffic_light_time-2dbox':{      'id':30,      'train_id':23,      'type_id':9   },  # 读秒灯
    'traffic_light_arrow-2dbox':{     'id':31,      'train_id':24,      'type_id':9   },  # 箭头灯
    'traffic_light_other-2dbox':{     'id':32,      'train_id':25,      'type_id':9   },  # 其他
    'vehicle_wheel-2dbox':{           'id':33,      'train_id':26,      'type_id':10  },  # 车_车轮
    'vehicle_light_on-2dbox':{        'id':34,      'train_id':27,      'type_id':11  },  # 车_车灯_亮
    'vehicle_light_off-2dbox':{       'id':35,      'train_id':28,      'type_id':11  },  # 车_车灯_不亮
    'rodcharge_genral-2dbox':{        'id':36,      'train_id':-1,      'type_id':12  },  # 收费杆
    'vehicle_car_head-2dbox':{        'id':37,      'train_id':29,      'type_id':13  },  # 小汽车_头
    'vehicle_car_rear-2dbox':{        'id':38,      'train_id':30,      'type_id':13  },  # 小汽车_尾
    'vehicle_bus_head-2dbox':{        'id':39,      'train_id':31,      'type_id':13  },  # 大巴车_头
    'vehicle_bus_rear-2dbox':{        'id':40,      'train_id':32,      'type_id':13  },  # 大巴车_尾
    'vehicle_truck_head-2dbox':{      'id':41,      'train_id':33,      'type_id':13  },  # 卡车_头
    'vehicle_truck_rear-2dbox':{      'id':42,      'train_id':34,      'type_id':13  },  # 卡车_尾
    'vehicle_tricycle_head-2dbox':{   'id':43,      'train_id':35,      'type_id':13  },  # 三轮车_头
    'vehicle_tricycle_rear-2dbox':{   'id':44,      'train_id':36,      'type_id':13  },  # 三轮车_尾
    'vehicle_else_head-2dbox':{       'id':45,      'train_id':37,      'type_id':13  },  # 异形车辆_头
    'vehicle_else_rear-2dbox':{       'id':46,      'train_id':38,      'type_id':13  },  # 异形车辆_尾
}

# 语义分割
SEMANTICS_DICT = {
    #    Name                                  ID             TrainID             TypeID
    'unlabeled':{                           'id':0,        'train_id':0,       'type_id':0  },  #  背景
    'road_general-area':{                   'id':1,        'train_id':1,       'type_id':1  },  #  可行驶区域
    'vehicle_car-area':{                    'id':2,        'train_id':2,       'type_id':2  },  #  小汽车
    'vehicle_bus-area':{                    'id':3,        'train_id':2,       'type_id':2  },  #  大巴车
    'vehicle_truck-area':{                  'id':4,        'train_id':2,       'type_id':2  },  #  卡车
    'vehicle_tricycle-area':{               'id':5,        'train_id':2,       'type_id':2  },  #  三轮车
    'vehicle_ridern-area':{                 'id':6,        'train_id':3,       'type_id':2  },  #  骑行车（无人）
    'vehicle_ridery-area':{                 'id':7,        'train_id':3,       'type_id':2  },  #  骑行车（有人）
    'vehicle_else-area':{                   'id':8,        'train_id':2,       'type_id':2  },  #  异形车辆
    'pedestrian_general-area':{             'id':9,        'train_id':4,       'type_id':3  },  #  普通行人
    'pedestrian_else-area':{                'id':10,       'train_id':4,       'type_id':3  },  #  异形人
    'animal_general-area':{                 'id':11,       'train_id':0,       'type_id':4  },  #  动物
    'obstacle_cone-area':{                  'id':12,       'train_id':5,       'type_id':5  },  #  锥桶
    'obstacle_triangle-area':{              'id':13,       'train_id':5,       'type_id':5  },  #  紧急三角架
    'obstacle_waterhorse-area':{            'id':14,       'train_id':5,       'type_id':5  },  #  水马
    'obstacle_else-area':{                  'id':15,       'train_id':5,       'type_id':5  },  #  人工障碍物（其他）
    'obstacle_column-area':{                'id':16,       'train_id':5,       'type_id':5  },  #  警示柱子
    'marking_diversionline_white-area':{    'id':17,       'train_id':6,       'type_id':6  },  #  白色导流带
    'marking_diversionline_yellow-area':{   'id':18,       'train_id':7,       'type_id':6  },  #  黄色导流带
    'marking_speedbump-area':{              'id':19,       'train_id':0,       'type_id':6  },  #  凸减速带
    'marking_nostopgrid_white-area':{       'id':20,       'train_id':6,       'type_id':6  },  #  白色禁停网格区
    'marking_nostopgrid_yellow-area':{      'id':21,       'train_id':7,       'type_id':6  },  #  黄色禁停网格区
    'marking_general_white-area':{          'id':22,       'train_id':6,       'type_id':6  },  #  白色路面符号
    'marking_general_yellow-area':{         'id':23,       'train_id':7,       'type_id':6  },  #  黄色路面符号
    'marking_line_white-area':{             'id':24,       'train_id':8,       'type_id':6  },  #  白色车道线
    'marking_line_yellow-area':{            'id':25,       'train_id':9,       'type_id':6  },  #  黄色车道线
    'marking_pavement_white-area':{         'id':26,       'train_id':6,       'type_id':6  },  #  白色人行道
    'marking_pavement_yellow-area':{        'id':27,       'train_id':7,       'type_id':6  },  #  黄色人行道
    'marking_stop_white-area':{             'id':28,       'train_id':6,       'type_id':6  },  #  白色停止线
    'marking_stop_yellow-area':{            'id':29,       'train_id':7,       'type_id':6  },  #  黄色停止线
    'marking_parking_white-area':{          'id':30,       'train_id':6,       'type_id':6  },  #  白色停车线
    'marking_parking_yellow-area':{         'id':31,       'train_id':7,       'type_id':6  },  #  黄色停车线
    'rod_charge-area':{                     'id':32,       'train_id':0,       'type_id':7  },  #  收费杆
    'rod_longmen-area':{                    'id':33,       'train_id':0,       'type_id':8  },  #  龙门架
    'self_genral-area':{                    'id':34,       'train_id':1,       'type_id':9  },  #  车本身(前视摄像头可能在底部获取车本身)
    'curb_line-line':{                      'id':35,       'train_id':10,       'type_id':6  },  #  道路结构线 
}

# 车道线
LANE_DICT = {
    #         Name                      ID             TrainID               TypeID
    'marking_genral-line':{           'id':1,        'train_id':-1,       'type_id':4  },  # 通用杆状物线
    'curb_line-line':{                'id':2,        'train_id':-1,       'type_id':3  },  # 道路结构线 
    'line-single-solid-white':{       'id':3,        'train_id':1,        'type_id':1  },  # 实线-白色
    'line-single-dash-white':{        'id':4,        'train_id':2,        'type_id':1  },  # 虚线-白色
    'marking_pavement-line':{         'id':5,        'train_id':-1,       'type_id':3  },  # 人行横道_线
    'line-single-dash-yellow':{       'id':6,        'train_id':3,        'type_id':1  },  # 虚线-黄色
    'line-road-edge':{                'id':7,        'train_id':-1,       'type_id':2  },  # 路沿线
    'rod_longmen-line':{              'id':8,        'train_id':-1,       'type_id':4  },  # 龙门架_线
    'line-single-solid-yellow':{      'id':9,        'train_id':-1,       'type_id':1  },  # 实线-黄色
    'marking_stop-line':{             'id':10,        'train_id':4,       'type_id':3  },  # 停止线_线
    'marking_slow-line':{             'id':11,       'train_id':-1,       'type_id':3  },  # 减速带_线
    'line-dash-solid-white':{         'id':12,       'train_id':-1,       'type_id':2  },  # 虚实线-左边虚线右边实线-白
    'line-slow':{                     'id':13,       'train_id':5,        'type_id':2  },  # 半鱼骨线
    'line-solid-dash-white':{         'id':14,       'train_id':5,        'type_id':2  },  # 虚实线-左边实线右边虚线-白
    'line-double-solid-yellow':{      'id':15,       'train_id':5,        'type_id':1  },  # 实线-双黄
    'line-solid-wide':{               'id':16,       'train_id':5,        'type_id':2  },  # 宽虚线
    'line-fish-bone-solid':{          'id':17,       'train_id':5,        'type_id':2  },  # 鱼骨线-中线实
    'line-fish-bone-dash':{           'id':18,       'train_id':5,        'type_id':2  },  # 鱼骨线-中线虚
    'line-double-solid-white':{       'id':19,       'train_id':5,        'type_id':1  },  # 实线-双白
    'line-double-dash-white':{        'id':20,       'train_id':5,        'type_id':1  },  # 虚线-双白
    'line-solid-dash-yellow':{        'id':21,       'train_id':5,        'type_id':2  },  # 虚实线-左边实线右边虚线-黄
    'marking_speedbump-line':{        'id':22,       'train_id':-1,       'type_id':3  },  # 减速带_凸_线
    'line-double-dash-yellow':{       'id':23,       'train_id':5,        'type_id':1  },  # 虚线-双黄
    'rod_charge-line':{               'id':24,       'train_id':-1,       'type_id':4  },  # 收费杆_线
    'line-dash-solid-yellow':{        'id':25,       'train_id':5,        'type_id':2  },  # 虚实线-左边虚线右边实线-黄
    'rod_genral-line':{               'id':26,       'train_id':-1,       'type_id':2  },  # 虚实线-左边虚线右边实线-黄
}


def _categorys_count_init(categorys_dict:dict = OBJECT_DICT) -> dict:
    '''
    初始化类别统计
    '''
    categorys_count = {}
    for key in categorys_dict.keys():
        categorys_count[key] = 0
    return categorys_count


def CA_multi_task_label(input_path: str, 
                        output_path: str, 
                        count_output_file: str, 
                        image_path: str='images', 
                        labels_lane: str='labels_lane', 
                        labels_obj: str='labels_obj', 
                        labels_semantic: str='labels_semantic') -> None:
    '''
    CA数据集的多任务联合标注标签处理,输出的标签包括目标检测、语义分割和车道线识别标签

    Args:
        input_path: 整体数据集保存的路径(由于CA的数据集前期表述非常的不一致,这里的读取方式相对而言比较奇葩)
        output_path: 标签输出的目录,最后包括目标检测、语义分割和车道线识别标签
        count_output_file: 计数文件保存的绝对路径,该文件用于统计各个标签的个数
        image_path: output_path路径下保存图像的目录名
        labels_lane: output_path路径下保存车道线标签的目录名
        labels_obj: output_path路径下保存目标检测标签的目录名
        labels_semantic: output_path路径下保存语义分割标签的目录名

    Returns:
        None
    '''
    input_path = Path(input_path).resolve()
    if not input_path.is_dir():
        raise Exception(f'ERROR: {str(input_path)} is not a dir')

    # 创建路径
    output_path = Path(output_path).resolve()
    image_path = output_path.joinpath(image_path)
    lane_path = output_path.joinpath(labels_lane)
    obj_path = output_path.joinpath(labels_obj)
    semantic_path = output_path.joinpath(labels_semantic)

    image_path.mkdir(exist_ok=True)
    lane_path.mkdir(exist_ok=True)
    obj_path.mkdir(exist_ok=True)
    semantic_path.mkdir(exist_ok=True)

    # 图像和标签搜寻
    files = sorted(glob.glob(str(input_path.joinpath(input_path, '**')), recursive=True))
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS] # 图像路径
    labels = [x for x in files if x.split('.')[-1].lower() in LABEL_FORMATS] # 标签路径
    
    # 图像处理
    images_dict = {}
    for x in tqdm(images, desc='Processing the image name and path!', unit='bathcs'):
        x = Path(x)
        parent_dir_last = x.parent.name.split('_')[-1] # 父目录最后'_'分割后的摄像头名称
        if 'cam' in parent_dir_last:
            stem_name = parent_dir_last + '_' + x.name
        else:
            stem_name = x.name
        image_name = image_path.joinpath(stem_name)
        object_name = obj_path.joinpath(stem_name.rsplit('.', 1)[0] + '.txt')
        lane_name = lane_path.joinpath(stem_name.rsplit('.', 1)[0] + '.json')
        semantic_name = semantic_path.joinpath(stem_name.rsplit('.', 1)[0] + '.png')
        images_dict.update({stem_name:[str(x), str(image_name), str(object_name), str(semantic_name), str(lane_name)]}) # 修改后的唯一名称: 原始路径名称,复制路径名称,目标标签名称,处理语义分割图名称,处理车道线名称

    # 标签处理
    label_dict = {}
    for x in tqdm(labels, desc='Fusing three kinds of label to one!', unit='labels'):
        with open(x, 'r', encoding='utf-8') as f:
            for oneline in f.readlines():
                if  oneline.startswith('http'):
                    url_path, _, label_json = oneline.split(maxsplit=2) # 百度标注的路径, 文件名称, 标签信息
                    label_json = json.loads(label_json)
                    url_path = Path(url_path)
                    parent_dir_last = url_path.parent.name.split('_')[-1]   
                    if 'cam' in parent_dir_last:
                        stem_name = parent_dir_last + '_' + url_path.name
                    else:
                        stem_name = url_path.name
                    if stem_name not in label_dict.keys():
                        label_dict.update({stem_name:label_json})
                    else:
                        label_dict[stem_name]['result'][0]['elements'] = label_dict[stem_name]['result'][0]['elements'] + label_json['result'][0]['elements']
    
    # 整体数据处理
    object_count = _categorys_count_init(OBJECT_DICT)
    semantics_count = _categorys_count_init(SEMANTICS_DICT)
    lane_count = _categorys_count_init(LANE_DICT)
    
    # TODO 多进程提升效率
    for x in tqdm(label_dict.keys(), desc='Copying the image and Processing the corresponding task label!', unit='imgs'):
        if x in images_dict.keys(): # 存在标签不一致的情况
            elements = label_dict[x]['result'][0]['elements']
            width = label_dict[x]['result'][0]['size']['width']
            height = label_dict[x]['result'][0]['size']['height']
            object_str = ''
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            lane_dict = {'imageHeight': height, 'imageWidth': width, 'shapes':[]} #! 倪光一数据格式定义
            for label in elements:
                for label_name in label['attribute'].values(): # 这里的属性值标注存在很多,名称只是其中之一
                    # 目标检测
                    if label_name in OBJECT_DICT.keys() and label['markType']=='rect': # 确保标注的是矩形框
                        train_id = OBJECT_DICT[label_name]['train_id'] #^ 获取train id
                        object_count[label_name] += 1
                        if train_id != -1:
                            w = label['width'] / width
                            h = label['height'] / height
                            x_central = label['posX'] / width + w / 2
                            y_central = label['posY'] / height + h / 2
                            object_str = object_str + str(train_id) + ' ' + ' '.join(("%.6f"%x_central, "%.6f"%y_central, "%.6f"%w, "%.6f"%h)) + '\n' #^ YOLOv5格式
                    # 语义分割
                    elif label_name in SEMANTICS_DICT.keys() and label['markType']=='area': # 确保标注的是区域
                        train_id = SEMANTICS_DICT[label_name]['train_id']
                        semantics_count[label_name] += 1
                        if train_id != -1:
                            xy = [(point['x'], point['y']) for point in label['points']]
                            assert len(xy) > 2, "Semantics must have points more than 2"
                            draw.polygon(xy, outline=1, fill=train_id)
                    # TODO 车道线(当前车道线采用的数据格式是倪光一自定义格式,没有管理trainid的相关情况后续迭代修改)
                    elif label_name in LANE_DICT.keys() and label['markType']=='line': # 确保标注的是线
                        lane_count[label_name] += 1
                        lane_dict['shapes'].append({'type':label_name, 'points':label['points']})
            # 分别写入数据
            # 复制图像
            shutil.copy(images_dict[x][0], images_dict[x][1])
            # YOLOv5标签
            with open(images_dict[x][2], 'w', encoding='utf-8') as f:
                f.write(object_str)
            # 语义分割PNG
            mask.save(images_dict[x][3])
            # 车道线格式转换
            with open(images_dict[x][4], 'w', encoding='utf-8') as f:
                json.dump(lane_dict, f)
        
    # 统计数量输出
    logging.info('Outputing the number statistics!')
    with open(count_output_file, 'w', encoding='utf-8') as f:
        f.write('目标检测统计\n')
        for key in sorted(object_count.keys()):
            f.write(key + ':' + str(object_count[key]) + '\n')
        f.write('\n语义分割统计\n')
        for key in sorted(semantics_count.keys()):
            f.write(key + ':' + str(semantics_count[key]) + '\n')
        f.write('\n车道线统计\n')
        for key in sorted(lane_count.keys()):
            f.write(key + ':' + str(lane_count[key]) + '\n')
            
    logging.info('All Finish! (*╹▽╹*),HaHa~')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA Joint MultiTask dataset format to corresponding task dataset format!", epilog="CA dataset is too complex! FUCK IT!")
    # parser.add_argument('-p', '--input_path', type=str, required=True, help='The whole dataset dir', metavar='CA_input_path')
    parser.add_argument('-p', '--input_path', type=str, default='./', help='The whole dataset dir', metavar='CA_input_path') # FOR IDE
    parser.add_argument('-o', '--output_path', type=str, default='./', help='The output dataset dir', metavar='CA_output_path')
    parser.add_argument('-c', '--count_output_file', type=str, default='./statistics.txt', help='The output name to save quantity statistics infomation', metavar='CA_count_output_path')
    opt = parser.parse_args()

    CA_multi_task_label(opt.input_path, opt.output_path, opt.count_output_file)