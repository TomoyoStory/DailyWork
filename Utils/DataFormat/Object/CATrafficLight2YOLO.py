# CA目标检测数据集格式转换为YOLO数据集格式脚本
# 由于标注的样式不一致，选取的数据处理程序根据情况选择

import json
import logging
import argparse
from tqdm import tqdm
from typing import Union
from pathlib import Path
import xml.etree.ElementTree as ET

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


#^ 以下定义根据CA标注需求和识别需求说明书定义，相关定义修改需严格遵照实际的需求和相关定义修改(常年根据实际情况改变)
# ID代表实例类别，TrainID代表参与训练的ID(-1代表不参与训练)，TypeID表示类别ID

# 车灯检测处理后标签定义
TRAFFIC_LIGHT_DICT = {
    #       Name                   ID             TrainID             TypeID
    'red_number_none':{          'id':0,       'train_id':0,       'type_id':4   },  # 红色数字
    'green_number_none':{        'id':1,       'train_id':1,       'type_id':5   },  # 绿色数字
    'yellow_number_none':{       'id':2,       'train_id':2,       'type_id':6   },  # 黄色数字
    'red_circle_none':{          'id':3,       'train_id':3,       'type_id':1   },  # 红灯
    'red_arrow_up':{             'id':4,       'train_id':4,       'type_id':2   },  # 红上
    'red_arrow_right':{          'id':5,       'train_id':5,       'type_id':2   },  # 红右
    'red_arrow_left':{           'id':6,       'train_id':6,       'type_id':2   },  # 红左
    'red_person_none':{          'id':7,       'train_id':7,       'type_id':3   },  # 红色
    'green_circle_none':{        'id':8,       'train_id':8,       'type_id':1   },  # 绿灯
    'green_arrow_up':{           'id':9,       'train_id':9,       'type_id':2   },  # 绿上
    'green_arrow_right':{        'id':10,      'train_id':10,      'type_id':2   },  # 绿右
    'green_arrow_left':{         'id':11,      'train_id':11,      'type_id':2   },  # 绿左
    'green_person_none':{        'id':12,      'train_id':12,      'type_id':3   },  # 绿色
    'yellow_circle_none':{       'id':13,      'train_id':13,      'type_id':1   },  # 黄灯
    'yellow_arrow_up':{          'id':14,      'train_id':14,      'type_id':2   },  # 黄上
    'yellow_arrow_right':{       'id':15,      'train_id':15,      'type_id':2   },  # 黄右
    'yellow_arrow_left':{        'id':16,      'train_id':16,      'type_id':2   },  # 黄左
    'yellow_person_none':{       'id':17,      'train_id':17,      'type_id':3   },  # 黄色
}

# 车灯检测原始标签定义
TRAFFIC_LIGHT_ORIGIN = {
    #       Name                   ID             TrainID             TypeID
    'red_circle_none':{         'id':1,        'train_id':3,       'type_id':1   },  # 红灯
    'green_circle_none':{       'id':2,        'train_id':8,       'type_id':1   },  # 绿灯
    'yellow_circle_none':{      'id':3,        'train_id':13,      'type_id':1   },  # 黄灯
    'red_arrow_up':{            'id':4,        'train_id':4,       'type_id':2   },  # 红上
    'red_arrow_right':{         'id':5,        'train_id':5,       'type_id':2   },  # 红右
    'red_arrow_left':{          'id':6,        'train_id':6,       'type_id':2   },  # 红左
    'red_arrow_down':{          'id':7,        'train_id':-1,      'type_id':2   },  # 红下
    'green_arrow_up':{          'id':8,        'train_id':9,       'type_id':2   },  # 绿上
    'green_arrow_right':{       'id':9,        'train_id':10,      'type_id':2   },  # 绿右
    'green_arrow_left':{        'id':10,       'train_id':11,      'type_id':2   },  # 绿左
    'green_arrow_down':{        'id':11,       'train_id':-1,      'type_id':2   },  # 绿下
    'yellow_arrow_up':{         'id':12,       'train_id':14,      'type_id':2   },  # 黄上
    'yellow_arrow_right':{      'id':13,       'train_id':15,      'type_id':2   },  # 黄右
    'yellow_arrow_left':{       'id':14,       'train_id':16,      'type_id':2   },  # 黄左
    'yellow_arrow_down':{       'id':15,       'train_id':-1,      'type_id':2   },  # 黄下
    'red_person_none':{         'id':16,       'train_id':7,       'type_id':3   },  # 红色
    'green_person_none':{       'id':17,       'train_id':12,      'type_id':3   },  # 绿色
    'yellow_person_none':{      'id':18,       'train_id':17,      'type_id':3   },  # 黄色
    'red_1_none':{              'id':19,       'train_id':0,       'type_id':4   },  # 红灯1
    'red_2_none':{              'id':20,       'train_id':0,       'type_id':4   },  # 红灯2
    'red_3_none':{              'id':21,       'train_id':0,       'type_id':4   },  # 红灯3
    'red_4_none':{              'id':22,       'train_id':0,       'type_id':4   },  # 红灯4
    'red_5_none':{              'id':23,       'train_id':0,       'type_id':4   },  # 红灯5
    'red_6_none':{              'id':24,       'train_id':0,       'type_id':4   },  # 红灯6
    'red_7_none':{              'id':25,       'train_id':0,       'type_id':4   },  # 红灯7
    'red_8_none':{              'id':26,       'train_id':0,       'type_id':4   },  # 红灯8
    'red_9_none':{              'id':27,       'train_id':0,       'type_id':4   },  # 红灯9
    'red_10_none':{             'id':28,       'train_id':0,       'type_id':4   },  # 红灯10
    'red_11_none':{             'id':29,       'train_id':0,       'type_id':4   },  # 红灯11
    'red_12_none':{             'id':30,       'train_id':0,       'type_id':4   },  # 红灯12
    'red_13_none':{             'id':31,       'train_id':0,       'type_id':4   },  # 红灯13
    'red_14_none':{             'id':32,       'train_id':0,       'type_id':4   },  # 红灯14
    'red_15_none':{             'id':33,       'train_id':0,       'type_id':4   },  # 红灯15
    'red_16_none':{             'id':34,       'train_id':0,       'type_id':4   },  # 红灯16
    'red_17_none':{             'id':35,       'train_id':0,       'type_id':4   },  # 红灯17
    'red_18_none':{             'id':36,       'train_id':0,       'type_id':4   },  # 红灯18
    'red_19_none':{             'id':37,       'train_id':0,       'type_id':4   },  # 红灯19
    'red_20_none':{             'id':38,       'train_id':0,       'type_id':4   },  # 红灯20
    'red_21_none':{             'id':39,       'train_id':0,       'type_id':4   },  # 红灯21
    'red_22_none':{             'id':40,       'train_id':0,       'type_id':4   },  # 红灯22
    'red_23_none':{             'id':41,       'train_id':0,       'type_id':4   },  # 红灯23
    'red_24_none':{             'id':42,       'train_id':0,       'type_id':4   },  # 红灯24
    'red_25_none':{             'id':43,       'train_id':0,       'type_id':4   },  # 红灯25
    'red_26_none':{             'id':44,       'train_id':0,       'type_id':4   },  # 红灯26
    'red_27_none':{             'id':45,       'train_id':0,       'type_id':4   },  # 红灯27
    'red_28_none':{             'id':46,       'train_id':0,       'type_id':4   },  # 红灯28
    'red_29_none':{             'id':47,       'train_id':0,       'type_id':4   },  # 红灯29
    'red_30_none':{             'id':48,       'train_id':0,       'type_id':4   },  # 红灯30
    'red_31_none':{             'id':49,       'train_id':0,       'type_id':4   },  # 红灯31
    'red_32_none':{             'id':50,       'train_id':0,       'type_id':4   },  # 红灯32
    'red_33_none':{             'id':51,       'train_id':0,       'type_id':4   },  # 红灯33
    'red_34_none':{             'id':52,       'train_id':0,       'type_id':4   },  # 红灯34
    'red_35_none':{             'id':53,       'train_id':0,       'type_id':4   },  # 红灯35
    'red_36_none':{             'id':54,       'train_id':0,       'type_id':4   },  # 红灯36
    'red_37_none':{             'id':55,       'train_id':0,       'type_id':4   },  # 红灯37
    'red_38_none':{             'id':56,       'train_id':0,       'type_id':4   },  # 红灯38
    'red_39_none':{             'id':57,       'train_id':0,       'type_id':4   },  # 红灯39
    'red_40_none':{             'id':58,       'train_id':0,       'type_id':4   },  # 红灯40
    'red_41_none':{             'id':59,       'train_id':0,       'type_id':4   },  # 红灯41
    'red_42_none':{             'id':60,       'train_id':0,       'type_id':4   },  # 红灯42
    'red_43_none':{             'id':61,       'train_id':0,       'type_id':4   },  # 红灯43
    'red_44_none':{             'id':62,       'train_id':0,       'type_id':4   },  # 红灯44
    'red_45_none':{             'id':63,       'train_id':0,       'type_id':4   },  # 红灯45
    'red_46_none':{             'id':64,       'train_id':0,       'type_id':4   },  # 红灯46
    'red_47_none':{             'id':65,       'train_id':0,       'type_id':4   },  # 红灯47
    'red_48_none':{             'id':66,       'train_id':0,       'type_id':4   },  # 红灯48
    'red_49_none':{             'id':67,       'train_id':0,       'type_id':4   },  # 红灯49
    'red_50_none':{             'id':68,       'train_id':0,       'type_id':4   },  # 红灯50
    'red_51_none':{             'id':69,       'train_id':0,       'type_id':4   },  # 红灯51
    'red_52_none':{             'id':70,       'train_id':0,       'type_id':4   },  # 红灯52
    'red_53_none':{             'id':71,       'train_id':0,       'type_id':4   },  # 红灯53
    'red_54_none':{             'id':72,       'train_id':0,       'type_id':4   },  # 红灯54
    'red_55_none':{             'id':73,       'train_id':0,       'type_id':4   },  # 红灯55
    'red_56_none':{             'id':74,       'train_id':0,       'type_id':4   },  # 红灯56
    'red_57_none':{             'id':75,       'train_id':0,       'type_id':4   },  # 红灯57
    'red_58_none':{             'id':76,       'train_id':0,       'type_id':4   },  # 红灯58
    'red_59_none':{             'id':77,       'train_id':0,       'type_id':4   },  # 红灯59
    'red_60_none':{             'id':78,       'train_id':0,       'type_id':4   },  # 红灯60
    'red_61_none':{             'id':79,       'train_id':0,       'type_id':4   },  # 红灯61
    'red_62_none':{             'id':80,       'train_id':0,       'type_id':4   },  # 红灯62
    'red_63_none':{             'id':81,       'train_id':0,       'type_id':4   },  # 红灯63
    'red_64_none':{             'id':82,       'train_id':0,       'type_id':4   },  # 红灯64
    'red_65_none':{             'id':83,       'train_id':0,       'type_id':4   },  # 红灯65
    'red_66_none':{             'id':84,       'train_id':0,       'type_id':4   },  # 红灯66
    'red_67_none':{             'id':85,       'train_id':0,       'type_id':4   },  # 红灯67
    'red_68_none':{             'id':86,       'train_id':0,       'type_id':4   },  # 红灯68
    'red_69_none':{             'id':87,       'train_id':0,       'type_id':4   },  # 红灯69
    'red_70_none':{             'id':88,       'train_id':0,       'type_id':4   },  # 红灯70
    'red_71_none':{             'id':89,       'train_id':0,       'type_id':4   },  # 红灯71
    'red_72_none':{             'id':90,       'train_id':0,       'type_id':4   },  # 红灯72
    'red_73_none':{             'id':91,       'train_id':0,       'type_id':4   },  # 红灯73
    'red_74_none':{             'id':92,       'train_id':0,       'type_id':4   },  # 红灯74
    'red_75_none':{             'id':93,       'train_id':0,       'type_id':4   },  # 红灯75
    'red_76_none':{             'id':94,       'train_id':0,       'type_id':4   },  # 红灯76
    'red_77_none':{             'id':95,       'train_id':0,       'type_id':4   },  # 红灯77
    'red_78_none':{             'id':96,       'train_id':0,       'type_id':4   },  # 红灯78
    'red_79_none':{             'id':97,       'train_id':0,       'type_id':4   },  # 红灯79
    'red_80_none':{             'id':98,       'train_id':0,       'type_id':4   },  # 红灯80
    'red_81_none':{             'id':99,       'train_id':0,       'type_id':4   },  # 红灯81
    'red_82_none':{             'id':100,      'train_id':0,       'type_id':4   },  # 红灯82
    'red_83_none':{             'id':101,      'train_id':0,       'type_id':4   },  # 红灯83
    'red_84_none':{             'id':102,      'train_id':0,       'type_id':4   },  # 红灯84
    'red_85_none':{             'id':103,      'train_id':0,       'type_id':4   },  # 红灯85
    'red_86_none':{             'id':104,      'train_id':0,       'type_id':4   },  # 红灯86
    'red_87_none':{             'id':105,      'train_id':0,       'type_id':4   },  # 红灯87
    'red_88_none':{             'id':106,      'train_id':0,       'type_id':4   },  # 红灯88
    'red_89_none':{             'id':107,      'train_id':0,       'type_id':4   },  # 红灯89
    'red_90_none':{             'id':108,      'train_id':0,       'type_id':4   },  # 红灯90
    'red_91_none':{             'id':109,      'train_id':0,       'type_id':4   },  # 红灯91
    'red_92_none':{             'id':110,      'train_id':0,       'type_id':4   },  # 红灯92
    'red_93_none':{             'id':111,      'train_id':0,       'type_id':4   },  # 红灯93
    'red_94_none':{             'id':112,      'train_id':0,       'type_id':4   },  # 红灯94
    'red_95_none':{             'id':113,      'train_id':0,       'type_id':4   },  # 红灯95
    'red_96_none':{             'id':114,      'train_id':0,       'type_id':4   },  # 红灯96
    'red_97_none':{             'id':115,      'train_id':0,       'type_id':4   },  # 红灯97
    'red_98_none':{             'id':116,      'train_id':0,       'type_id':4   },  # 红灯98
    'red_99_none':{             'id':117,      'train_id':0,       'type_id':4   },  # 红灯99
    'green_1_none':{            'id':118,      'train_id':1,       'type_id':5   },  # 绿灯1
    'green_2_none':{            'id':119,      'train_id':1,       'type_id':5   },  # 绿灯2
    'green_3_none':{            'id':120,      'train_id':1,       'type_id':5   },  # 绿灯3
    'green_4_none':{            'id':121,      'train_id':1,       'type_id':5   },  # 绿灯4
    'green_5_none':{            'id':122,      'train_id':1,       'type_id':5   },  # 绿灯5
    'green_6_none':{            'id':123,      'train_id':1,       'type_id':5   },  # 绿灯6
    'green_7_none':{            'id':124,      'train_id':1,       'type_id':5   },  # 绿灯7
    'green_8_none':{            'id':125,      'train_id':1,       'type_id':5   },  # 绿灯8
    'green_9_none':{            'id':126,      'train_id':1,       'type_id':5   },  # 绿灯9
    'green_10_none':{           'id':127,      'train_id':1,       'type_id':5   },  # 绿灯10
    'green_11_none':{           'id':128,      'train_id':1,       'type_id':5   },  # 绿灯11
    'green_12_none':{           'id':129,      'train_id':1,       'type_id':5   },  # 绿灯12
    'green_13_none':{           'id':130,      'train_id':1,       'type_id':5   },  # 绿灯13
    'green_14_none':{           'id':131,      'train_id':1,       'type_id':5   },  # 绿灯14
    'green_15_none':{           'id':132,      'train_id':1,       'type_id':5   },  # 绿灯15
    'green_16_none':{           'id':133,      'train_id':1,       'type_id':5   },  # 绿灯16
    'green_17_none':{           'id':134,      'train_id':1,       'type_id':5   },  # 绿灯17
    'green_18_none':{           'id':135,      'train_id':1,       'type_id':5   },  # 绿灯18
    'green_19_none':{           'id':136,      'train_id':1,       'type_id':5   },  # 绿灯19
    'green_20_none':{           'id':137,      'train_id':1,       'type_id':5   },  # 绿灯20
    'green_21_none':{           'id':138,      'train_id':1,       'type_id':5   },  # 绿灯21
    'green_22_none':{           'id':139,      'train_id':1,       'type_id':5   },  # 绿灯22
    'green_23_none':{           'id':140,      'train_id':1,       'type_id':5   },  # 绿灯23
    'green_24_none':{           'id':141,      'train_id':1,       'type_id':5   },  # 绿灯24
    'green_25_none':{           'id':142,      'train_id':1,       'type_id':5   },  # 绿灯25
    'green_26_none':{           'id':143,      'train_id':1,       'type_id':5   },  # 绿灯26
    'green_27_none':{           'id':144,      'train_id':1,       'type_id':5   },  # 绿灯27
    'green_28_none':{           'id':145,      'train_id':1,       'type_id':5   },  # 绿灯28
    'green_29_none':{           'id':146,      'train_id':1,       'type_id':5   },  # 绿灯29
    'green_30_none':{           'id':147,      'train_id':1,       'type_id':5   },  # 绿灯30
    'green_31_none':{           'id':148,      'train_id':1,       'type_id':5   },  # 绿灯31
    'green_32_none':{           'id':149,      'train_id':1,       'type_id':5   },  # 绿灯32
    'green_33_none':{           'id':150,      'train_id':1,       'type_id':5   },  # 绿灯33
    'green_34_none':{           'id':151,      'train_id':1,       'type_id':5   },  # 绿灯34
    'green_35_none':{           'id':152,      'train_id':1,       'type_id':5   },  # 绿灯35
    'green_36_none':{           'id':153,      'train_id':1,       'type_id':5   },  # 绿灯36
    'green_37_none':{           'id':154,      'train_id':1,       'type_id':5   },  # 绿灯37
    'green_38_none':{           'id':155,      'train_id':1,       'type_id':5   },  # 绿灯38
    'green_39_none':{           'id':156,      'train_id':1,       'type_id':5   },  # 绿灯39
    'green_40_none':{           'id':157,      'train_id':1,       'type_id':5   },  # 绿灯40
    'green_41_none':{           'id':158,      'train_id':1,       'type_id':5   },  # 绿灯41
    'green_42_none':{           'id':159,      'train_id':1,       'type_id':5   },  # 绿灯42
    'green_43_none':{           'id':160,      'train_id':1,       'type_id':5   },  # 绿灯43
    'green_44_none':{           'id':161,      'train_id':1,       'type_id':5   },  # 绿灯44
    'green_45_none':{           'id':162,      'train_id':1,       'type_id':5   },  # 绿灯45
    'green_46_none':{           'id':163,      'train_id':1,       'type_id':5   },  # 绿灯46
    'green_47_none':{           'id':164,      'train_id':1,       'type_id':5   },  # 绿灯47
    'green_48_none':{           'id':165,      'train_id':1,       'type_id':5   },  # 绿灯48
    'green_49_none':{           'id':166,      'train_id':1,       'type_id':5   },  # 绿灯49
    'green_50_none':{           'id':167,      'train_id':1,       'type_id':5   },  # 绿灯50
    'green_51_none':{           'id':168,      'train_id':1,       'type_id':5   },  # 绿灯51
    'green_52_none':{           'id':169,      'train_id':1,       'type_id':5   },  # 绿灯52
    'green_53_none':{           'id':170,      'train_id':1,       'type_id':5   },  # 绿灯53
    'green_54_none':{           'id':171,      'train_id':1,       'type_id':5   },  # 绿灯54
    'green_55_none':{           'id':172,      'train_id':1,       'type_id':5   },  # 绿灯55
    'green_56_none':{           'id':173,      'train_id':1,       'type_id':5   },  # 绿灯56
    'green_57_none':{           'id':174,      'train_id':1,       'type_id':5   },  # 绿灯57
    'green_58_none':{           'id':175,      'train_id':1,       'type_id':5   },  # 绿灯58
    'green_59_none':{           'id':176,      'train_id':1,       'type_id':5   },  # 绿灯59
    'green_60_none':{           'id':177,      'train_id':1,       'type_id':5   },  # 绿灯60
    'green_61_none':{           'id':178,      'train_id':1,       'type_id':5   },  # 绿灯61
    'green_62_none':{           'id':179,      'train_id':1,       'type_id':5   },  # 绿灯62
    'green_63_none':{           'id':180,      'train_id':1,       'type_id':5   },  # 绿灯63
    'green_64_none':{           'id':181,      'train_id':1,       'type_id':5   },  # 绿灯64
    'green_65_none':{           'id':182,      'train_id':1,       'type_id':5   },  # 绿灯65
    'green_66_none':{           'id':183,      'train_id':1,       'type_id':5   },  # 绿灯66
    'green_67_none':{           'id':184,      'train_id':1,       'type_id':5   },  # 绿灯67
    'green_68_none':{           'id':185,      'train_id':1,       'type_id':5   },  # 绿灯68
    'green_69_none':{           'id':186,      'train_id':1,       'type_id':5   },  # 绿灯69
    'green_70_none':{           'id':187,      'train_id':1,       'type_id':5   },  # 绿灯70
    'green_71_none':{           'id':188,      'train_id':1,       'type_id':5   },  # 绿灯71
    'green_72_none':{           'id':189,      'train_id':1,       'type_id':5   },  # 绿灯72
    'green_73_none':{           'id':190,      'train_id':1,       'type_id':5   },  # 绿灯73
    'green_74_none':{           'id':191,      'train_id':1,       'type_id':5   },  # 绿灯74
    'green_75_none':{           'id':192,      'train_id':1,       'type_id':5   },  # 绿灯75
    'green_76_none':{           'id':193,      'train_id':1,       'type_id':5   },  # 绿灯76
    'green_77_none':{           'id':194,      'train_id':1,       'type_id':5   },  # 绿灯77
    'green_78_none':{           'id':195,      'train_id':1,       'type_id':5   },  # 绿灯78
    'green_79_none':{           'id':196,      'train_id':1,       'type_id':5   },  # 绿灯79
    'green_80_none':{           'id':197,      'train_id':1,       'type_id':5   },  # 绿灯80
    'green_81_none':{           'id':198,      'train_id':1,       'type_id':5   },  # 绿灯81
    'green_82_none':{           'id':199,      'train_id':1,       'type_id':5   },  # 绿灯82
    'green_83_none':{           'id':200,      'train_id':1,       'type_id':5   },  # 绿灯83
    'green_84_none':{           'id':201,      'train_id':1,       'type_id':5   },  # 绿灯84
    'green_85_none':{           'id':202,      'train_id':1,       'type_id':5   },  # 绿灯85
    'green_86_none':{           'id':203,      'train_id':1,       'type_id':5   },  # 绿灯86
    'green_87_none':{           'id':204,      'train_id':1,       'type_id':5   },  # 绿灯87
    'green_88_none':{           'id':205,      'train_id':1,       'type_id':5   },  # 绿灯88
    'green_89_none':{           'id':206,      'train_id':1,       'type_id':5   },  # 绿灯89
    'green_90_none':{           'id':207,      'train_id':1,       'type_id':5   },  # 绿灯90
    'green_91_none':{           'id':208,      'train_id':1,       'type_id':5   },  # 绿灯91
    'green_92_none':{           'id':209,      'train_id':1,       'type_id':5   },  # 绿灯92
    'green_93_none':{           'id':210,      'train_id':1,       'type_id':5   },  # 绿灯93
    'green_94_none':{           'id':211,      'train_id':1,       'type_id':5   },  # 绿灯94
    'green_95_none':{           'id':212,      'train_id':1,       'type_id':5   },  # 绿灯95
    'green_96_none':{           'id':213,      'train_id':1,       'type_id':5   },  # 绿灯96
    'green_97_none':{           'id':214,      'train_id':1,       'type_id':5   },  # 绿灯97
    'green_98_none':{           'id':215,      'train_id':1,       'type_id':5   },  # 绿灯98
    'green_99_none':{           'id':216,      'train_id':1,       'type_id':5   },  # 绿灯99
    'yellow_1_none':{           'id':217,      'train_id':2,       'type_id':6   },  # 黄灯1
    'yellow_2_none':{           'id':218,      'train_id':2,       'type_id':6   },  # 黄灯2
    'yellow_3_none':{           'id':219,      'train_id':2,       'type_id':6   },  # 黄灯3
    'yellow_4_none':{           'id':220,      'train_id':2,       'type_id':6   },  # 黄灯4
    'yellow_5_none':{           'id':221,      'train_id':2,       'type_id':6   },  # 黄灯5
    'yellow_6_none':{           'id':222,      'train_id':2,       'type_id':6   },  # 黄灯6
    'yellow_7_none':{           'id':223,      'train_id':2,       'type_id':6   },  # 黄灯7
    'yellow_8_none':{           'id':224,      'train_id':2,       'type_id':6   },  # 黄灯8
    'yellow_9_none':{           'id':225,      'train_id':2,       'type_id':6   },  # 黄灯9
    'yellow_10_none':{          'id':226,      'train_id':2,       'type_id':6   },  # 黄灯10
    'yellow_11_none':{          'id':227,      'train_id':2,       'type_id':6   },  # 黄灯11
    'yellow_12_none':{          'id':228,      'train_id':2,       'type_id':6   },  # 黄灯12
    'yellow_13_none':{          'id':229,      'train_id':2,       'type_id':6   },  # 黄灯13
    'yellow_14_none':{          'id':230,      'train_id':2,       'type_id':6   },  # 黄灯14
    'yellow_15_none':{          'id':231,      'train_id':2,       'type_id':6   },  # 黄灯15
    'yellow_16_none':{          'id':232,      'train_id':2,       'type_id':6   },  # 黄灯16
    'yellow_17_none':{          'id':233,      'train_id':2,       'type_id':6   },  # 黄灯17
    'yellow_18_none':{          'id':234,      'train_id':2,       'type_id':6   },  # 黄灯18
    'yellow_19_none':{          'id':235,      'train_id':2,       'type_id':6   },  # 黄灯19
    'yellow_20_none':{          'id':236,      'train_id':2,       'type_id':6   },  # 黄灯20
    'yellow_21_none':{          'id':237,      'train_id':2,       'type_id':6   },  # 黄灯21
    'yellow_22_none':{          'id':238,      'train_id':2,       'type_id':6   },  # 黄灯22
    'yellow_23_none':{          'id':239,      'train_id':2,       'type_id':6   },  # 黄灯23
    'yellow_24_none':{          'id':240,      'train_id':2,       'type_id':6   },  # 黄灯24
    'yellow_25_none':{          'id':241,      'train_id':2,       'type_id':6   },  # 黄灯25
    'yellow_26_none':{          'id':242,      'train_id':2,       'type_id':6   },  # 黄灯26
    'yellow_27_none':{          'id':243,      'train_id':2,       'type_id':6   },  # 黄灯27
    'yellow_28_none':{          'id':244,      'train_id':2,       'type_id':6   },  # 黄灯28
    'yellow_29_none':{          'id':245,      'train_id':2,       'type_id':6   },  # 黄灯29
    'yellow_30_none':{          'id':246,      'train_id':2,       'type_id':6   },  # 黄灯30
    'yellow_31_none':{          'id':247,      'train_id':2,       'type_id':6   },  # 黄灯31
    'yellow_32_none':{          'id':248,      'train_id':2,       'type_id':6   },  # 黄灯32
    'yellow_33_none':{          'id':249,      'train_id':2,       'type_id':6   },  # 黄灯33
    'yellow_34_none':{          'id':250,      'train_id':2,       'type_id':6   },  # 黄灯34
    'yellow_35_none':{          'id':251,      'train_id':2,       'type_id':6   },  # 黄灯35
    'yellow_36_none':{          'id':252,      'train_id':2,       'type_id':6   },  # 黄灯36
    'yellow_37_none':{          'id':253,      'train_id':2,       'type_id':6   },  # 黄灯37
    'yellow_38_none':{          'id':254,      'train_id':2,       'type_id':6   },  # 黄灯38
    'yellow_39_none':{          'id':255,      'train_id':2,       'type_id':6   },  # 黄灯39
    'yellow_40_none':{          'id':256,      'train_id':2,       'type_id':6   },  # 黄灯40
    'yellow_41_none':{          'id':257,      'train_id':2,       'type_id':6   },  # 黄灯41
    'yellow_42_none':{          'id':258,      'train_id':2,       'type_id':6   },  # 黄灯42
    'yellow_43_none':{          'id':259,      'train_id':2,       'type_id':6   },  # 黄灯43
    'yellow_44_none':{          'id':260,      'train_id':2,       'type_id':6   },  # 黄灯44
    'yellow_45_none':{          'id':261,      'train_id':2,       'type_id':6   },  # 黄灯45
    'yellow_46_none':{          'id':262,      'train_id':2,       'type_id':6   },  # 黄灯46
    'yellow_47_none':{          'id':263,      'train_id':2,       'type_id':6   },  # 黄灯47
    'yellow_48_none':{          'id':264,      'train_id':2,       'type_id':6   },  # 黄灯48
    'yellow_49_none':{          'id':265,      'train_id':2,       'type_id':6   },  # 黄灯49
    'yellow_50_none':{          'id':266,      'train_id':2,       'type_id':6   },  # 黄灯50
    'yellow_51_none':{          'id':267,      'train_id':2,       'type_id':6   },  # 黄灯51
    'yellow_52_none':{          'id':268,      'train_id':2,       'type_id':6   },  # 黄灯52
    'yellow_53_none':{          'id':269,      'train_id':2,       'type_id':6   },  # 黄灯53
    'yellow_54_none':{          'id':270,      'train_id':2,       'type_id':6   },  # 黄灯54
    'yellow_55_none':{          'id':271,      'train_id':2,       'type_id':6   },  # 黄灯55
    'yellow_56_none':{          'id':272,      'train_id':2,       'type_id':6   },  # 黄灯56
    'yellow_57_none':{          'id':273,      'train_id':2,       'type_id':6   },  # 黄灯57
    'yellow_58_none':{          'id':274,      'train_id':2,       'type_id':6   },  # 黄灯58
    'yellow_59_none':{          'id':275,      'train_id':2,       'type_id':6   },  # 黄灯59
    'yellow_60_none':{          'id':276,      'train_id':2,       'type_id':6   },  # 黄灯60
    'yellow_61_none':{          'id':277,      'train_id':2,       'type_id':6   },  # 黄灯61
    'yellow_62_none':{          'id':278,      'train_id':2,       'type_id':6   },  # 黄灯62
    'yellow_63_none':{          'id':279,      'train_id':2,       'type_id':6   },  # 黄灯63
    'yellow_64_none':{          'id':280,      'train_id':2,       'type_id':6   },  # 黄灯64
    'yellow_65_none':{          'id':281,      'train_id':2,       'type_id':6   },  # 黄灯65
    'yellow_66_none':{          'id':282,      'train_id':2,       'type_id':6   },  # 黄灯66
    'yellow_67_none':{          'id':283,      'train_id':2,       'type_id':6   },  # 黄灯67
    'yellow_68_none':{          'id':284,      'train_id':2,       'type_id':6   },  # 黄灯68
    'yellow_69_none':{          'id':285,      'train_id':2,       'type_id':6   },  # 黄灯69
    'yellow_70_none':{          'id':286,      'train_id':2,       'type_id':6   },  # 黄灯70
    'yellow_71_none':{          'id':287,      'train_id':2,       'type_id':6   },  # 黄灯71
    'yellow_72_none':{          'id':288,      'train_id':2,       'type_id':6   },  # 黄灯72
    'yellow_73_none':{          'id':289,      'train_id':2,       'type_id':6   },  # 黄灯73
    'yellow_74_none':{          'id':290,      'train_id':2,       'type_id':6   },  # 黄灯74
    'yellow_75_none':{          'id':291,      'train_id':2,       'type_id':6   },  # 黄灯75
    'yellow_76_none':{          'id':292,      'train_id':2,       'type_id':6   },  # 黄灯76
    'yellow_77_none':{          'id':293,      'train_id':2,       'type_id':6   },  # 黄灯77
    'yellow_78_none':{          'id':294,      'train_id':2,       'type_id':6   },  # 黄灯78
    'yellow_79_none':{          'id':295,      'train_id':2,       'type_id':6   },  # 黄灯79
    'yellow_80_none':{          'id':296,      'train_id':2,       'type_id':6   },  # 黄灯80
    'yellow_81_none':{          'id':297,      'train_id':2,       'type_id':6   },  # 黄灯81
    'yellow_82_none':{          'id':298,      'train_id':2,       'type_id':6   },  # 黄灯82
    'yellow_83_none':{          'id':299,      'train_id':2,       'type_id':6   },  # 黄灯83
    'yellow_84_none':{          'id':300,      'train_id':2,       'type_id':6   },  # 黄灯84
    'yellow_85_none':{          'id':301,      'train_id':2,       'type_id':6   },  # 黄灯85
    'yellow_86_none':{          'id':302,      'train_id':2,       'type_id':6   },  # 黄灯86
    'yellow_87_none':{          'id':303,      'train_id':2,       'type_id':6   },  # 黄灯87
    'yellow_88_none':{          'id':304,      'train_id':2,       'type_id':6   },  # 黄灯88
    'yellow_89_none':{          'id':305,      'train_id':2,       'type_id':6   },  # 黄灯89
    'yellow_90_none':{          'id':306,      'train_id':2,       'type_id':6   },  # 黄灯90
    'yellow_91_none':{          'id':307,      'train_id':2,       'type_id':6   },  # 黄灯91
    'yellow_92_none':{          'id':308,      'train_id':2,       'type_id':6   },  # 黄灯92
    'yellow_93_none':{          'id':309,      'train_id':2,       'type_id':6   },  # 黄灯93
    'yellow_94_none':{          'id':310,      'train_id':2,       'type_id':6   },  # 黄灯94
    'yellow_95_none':{          'id':311,      'train_id':2,       'type_id':6   },  # 黄灯95
    'yellow_96_none':{          'id':312,      'train_id':2,       'type_id':6   },  # 黄灯96
    'yellow_97_none':{          'id':313,      'train_id':2,       'type_id':6   },  # 黄灯97
    'yellow_98_none':{          'id':314,      'train_id':2,       'type_id':6   },  # 黄灯98
    'yellow_99_none':{          'id':315,      'train_id':2,       'type_id':6   },  # 黄灯99
    'none_none_none':{          'id':316,      'train_id':-1,      'type_id':7   },  # 大框
    'none_small_none':{         'id':317,      'train_id':-1,      'type_id':7   },  # 小框
}


def _statistics_lables(count_output_file: str, traffic_light_count: dict) -> None:
    '''
    该函数用于提取统计各个统计标签的信息

    Args:
        count_output_file: 统计文件的绝对路径
        traffic_light_count: 统计的交通灯各类类别个数
    
    Returns:
        None
    '''
    count_output_file = Path(count_output_file)
    if count_output_file.exists():
        logging.warn(f'The {str(count_output_file)} exists, the new count number will be added to it!')
        with open(count_output_file, 'r', encoding='utf-8') as f:
            for x in f.readlines():
                label, count = x.split(':', 1)
                count = int(count)
                traffic_light_count[label] += count

    with open(count_output_file, 'w', encoding='utf-8') as f:
            for key in traffic_light_count.keys():
                f.write(key + ':' + str(traffic_light_count[key]) + '\n')

        
def CA_BAIDU_traffic_light_to_YOLO(lable_file: str,
                                   output_path: str,
                                   count_output_file: str) -> None:
    '''
    该函数用于提取CA标注的交通灯的标签信息，注意，这里的格式输出为x_center,y_center,w,h

    Args:
        lable_file: 标注的文档路径
        output_path: yolo格式输出的路径
        count_output_file: 输出的模型数据量统计
    
    Returns:
        None
    '''
    lable_file = Path(lable_file)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # 类别统计
    traffic_light_count = {}
    for key in TRAFFIC_LIGHT_DICT.keys():
        traffic_light_count[key] = 0
    traffic_light_count['other'] = 0

    logging.info('Writing YOLO format labels')
    with open(lable_file, 'r', encoding='utf-8') as f:
        for oneline in f.readlines():
            if  oneline.startswith('http'):
                url_path, _, label_json = oneline.split(maxsplit=2) # 百度标注的路径, 文件名称, 标签信息
                label_json = json.loads(label_json)
                url_path = Path(url_path)
                elements = label_json['result'][0]['elements']
                width = label_json['result'][0]['size']['width']
                height = label_json['result'][0]['size']['height']
                object_str = ''
                for label in elements:
                    for label_name in label['attribute'].values(): # 这里的属性值标注存在很多，名称只是其中之一
                        if label_name in TRAFFIC_LIGHT_ORIGIN.keys() and label['markType']=='rect': # 确保标注的是矩形框
                            train_id = TRAFFIC_LIGHT_ORIGIN[label_name]['train_id'] #! 通过原始定义获取TrainId
                            # 标签名称转换和数量统计 #^ 这里根据实际情况修改
                            if train_id==0:
                                traffic_light_count['red_number_none'] +=1
                            elif train_id==1:
                                traffic_light_count['green_number_none'] +=1
                            elif train_id==2:
                                traffic_light_count['yellow_number_none'] +=1
                            else:
                                try:
                                    traffic_light_count[label_name] += 1
                                except:
                                    traffic_light_count['other'] += 1

                            if train_id != -1:
                                w = label['width'] / width
                                h = label['height'] / height
                                x_central = label['posX'] / width + w / 2
                                y_central = label['posY'] / height + h / 2
                                object_str = object_str + str(train_id) + ' ' + ' '.join(("%.6f"%x_central, "%.6f"%y_central, "%.6f"%w, "%.6f"%h)) + '\n' #^ YOLOv5格式
                # 写YOLOv5标签
                with open(output_path.joinpath(url_path.stem + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(object_str)
    
    # 统计数量输出
    logging.info('Outputing the number statistics!')
    _statistics_lables(count_output_file, traffic_light_count)
    logging.info('All Finish! (*╹▽╹*), HaHa~')


def CA_LabelMe_VOC_to_YOLO(input_path: str, 
                           output_path: str, 
                           count_output_file: str,
                           width: Union[int, None] =None, 
                           height: Union[int, None] = None, 
                           save_difficult: bool=False ) -> None:
    '''
    将VOC的xml格式数据转换为YOLOv5的xywh格式

    Args:
        input_path: 输入xml标签的路径
        output_path: 输出txt标签的路径
        count_output_file: 输出的模型数据量统计
        width: 自行设置的覆盖内部XML宽高的宽
        height: 自行设置的覆盖内部XML宽高的高
        save_difficult: 是否保留难样例
        
    Returns:
        None
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    # 类别统计
    traffic_light_count = {}
    for key in TRAFFIC_LIGHT_DICT.keys():
        traffic_light_count[key] = 0
    traffic_light_count['other'] = 0

    logging.info('Getting the xml labels')
    images_path = list(input_path.glob('*.xml'))
    for label_file in tqdm(images_path, desc='Changing CA VOC format to YOLO format!', unit='xmls'):
        with open(output_path.joinpath(label_file.stem + '.txt'), 'w', encoding='utf-8') as out_file:
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text) # 宽
            h = int(size.find('height').text) # 高
            if width: # 长宽覆盖
                w = width
            if height:
                h = height
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                label_name = obj.find('name').text
                
                if save_difficult: # 是否过滤难样例
                    if label_name not in TRAFFIC_LIGHT_ORIGIN.keys():
                        continue
                else:
                    if label_name not in TRAFFIC_LIGHT_ORIGIN or int(difficult) == 1:
                        continue

                train_id = TRAFFIC_LIGHT_ORIGIN[label_name]['train_id']

                # 标签名称转换和数量统计 #^ 这里根据实际情况修改
                if train_id==0:
                    traffic_light_count['red_number_none'] +=1
                elif train_id==1:
                    traffic_light_count['green_number_none'] +=1
                elif train_id==2:
                    traffic_light_count['yellow_number_none'] +=1
                else:
                    try:
                        traffic_light_count[label_name] += 1
                    except:
                        traffic_light_count['other'] += 1

                if train_id != -1:
                    bndbox = obj.find('bndbox')
                    box = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)]
                    x_cnetral = (box[0] + box[2]) / (2 * w)
                    y_central = (box[1] + box[3]) / (2 * h)
                    w_ = (box[2] - box[0]) / w
                    h_ = (box[3] - box[1]) / h

                    out_file.write(str(train_id) + " " + " ".join('%.6f'%i for i in [x_cnetral, y_central, w_, h_]) + '\n') # 每次写入一行(思路与直接写还不太一致)
    
    # 统计数量输出
    logging.info('Outputing the number statistics!')
    _statistics_lables(count_output_file, traffic_light_count)
    logging.info('All Finish! (*╹▽╹*), HaHa~')


def CA_Single_Json_to_YOLO(input_path: str, 
                           output_path: str, 
                           count_output_file: str) -> None:
    '''
    将百度标注的单独的Json格式转换为YOLO格式

    Args:
        input_path: 输入json格式文件标签的路径
        output_path: 输出txt标签的路径
        count_output_file: 输出的模型数据量统计
        
    Returns:
        None
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # 类别统计
    traffic_light_count = {}
    for key in TRAFFIC_LIGHT_DICT.keys():
        traffic_light_count[key] = 0
    traffic_light_count['other'] = 0

    logging.info('Getting the json labels')
    images_path = list(input_path.glob('*.json'))

    for label_file in tqdm(images_path, desc='Changing CA BAIDU json format to YOLO format!', unit='jsons'):
        with open(label_file, 'r', encoding='utf-8') as f:
            json_str = json.load(f)
        width = json_str['imageWidth']
        height = json_str['imageHeight']
        label_str = ''
        for x in json_str['shapes']:
            label_name = x['label']
            train_id = TRAFFIC_LIGHT_ORIGIN[label_name]['train_id']
            # 标签名称转换和数量统计 #^ 这里根据实际情况修改
            if train_id==0:
                traffic_light_count['red_number_none'] +=1
            elif train_id==1:
                traffic_light_count['green_number_none'] +=1
            elif train_id==2:
                traffic_light_count['yellow_number_none'] +=1
            else:
                try:
                    traffic_light_count[label_name] += 1
                except:
                    traffic_light_count['other'] += 1

            if train_id != -1:
                x_cnetral = (x['points'][0][0] + x['points'][1][0]) / (2 * width)
                y_central = (x['points'][0][1] + x['points'][1][1]) / (2 * height)
                w = abs((x['points'][1][0] - x['points'][0][0])) / width
                h = abs((x['points'][1][1] - x['points'][0][1])) / height
                label_str = label_str + str(train_id) + ' ' + ' '.join('%.6f'%i for i in [x_cnetral, y_central, w, h]) + '\n'
            
        with open(output_path.joinpath(label_file.stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write(label_str)
    
    # 统计数量输出
    logging.info('Outputing the number statistics!')
    _statistics_lables(count_output_file, traffic_light_count)
    logging.info('All Finish! (*╹▽╹*), HaHa~')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA Traffic Light dataset format to Yolov5 format!", epilog="Just do it!")
    # parser.add_argument('-l', '--lable_file', type=str,  required=True, help='The label absolute path or absolute dir path', metavar='lable_file')
    parser.add_argument('-l', '--lable_file', type=str, default='./', help='The label absolute path or absolute dir path', metavar='lable_file') # For IDE
    parser.add_argument('-o', '--output_path', type=str, default='./labels', help='The output dataset dir', metavar='CA_output_path')
    parser.add_argument('-c', '--count_output_file', type=str, default='./Statistics.txt', help='The output name to save quantity statistics infomation', metavar='CA_count_output_path')
    opt = parser.parse_args()

    # CA_BAIDU_traffic_light_to_YOLO(opt.lable_file, opt.output_path, opt.count_output_file)

    # CA_LabelMe_VOC_to_YOLO(opt.lable_file, opt.output_path, opt.count_output_file, 1920, 1080) #^ 此处的长宽根据实际情况自行调整

    # CA_Single_Json_to_YOLO(opt.lable_file, opt.output_path, opt.count_output_file)
