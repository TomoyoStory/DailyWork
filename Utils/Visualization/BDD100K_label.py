# BDD Label definition.
# BDD数据集的标签定义
# 代码来自原版bdd的label.py文件
# 其他参考 https://doc.bdd100k.com/format.html

from collections import namedtuple

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        # 标签名称
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        # 标签定义ID
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        # 训练ID
        "category",  # The name of the category that this label belongs to
        # 属于的类别
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        # 类别ID
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        # 是否有实例区分
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        # 是否忽略评估
        "color",  # The color of this label
        # 对应颜色
    ],
)

# 全景分割类别定义
# Our extended list of label types. Our train id is compatible with Cityscapes
labels = [
    #           name                id    trainId   category          catId      hasInstances   ignoreInEval      color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    Label("guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    Label("traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    Label(
        "traffic sign frame",
        27,
        255,
        "object",
        3,
        False,
        True,
        (250, 170, 250),
    ),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]

# 可行驶区域标签定义
drivables = [
    #         name            id   trainId     category      catId    hasInstances    ignoreInEval       color
    Label("direct", 0, 0, "drivable", 0, False, False, (219, 94, 86)),
    Label("alternative", 1, 1, "drivable", 0, False, False, (86, 211, 219)),
    Label("background", 2, 2, "drivable", 0, False, False, (0, 0, 0)),
]

# 线型方向标签定义
lane_directions = [
    #         name         id    trainId     category     catId      hasInstances   ignoreInEval       color
    Label("parallel", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("vertical", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

# 线型风格标签定义
lane_styles = [
    #         name      id    trainId     category      catId      hasInstances   ignoreInEval      color
    Label("solid", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("dashed", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

# 线型类型标签定义
lane_categories = [
    #           name            id   trainId      category      catId      hasInstances       ignoreInEval          color
    Label("crosswalk", 0, 0, "lane_mark", 0, False, False, (219, 94, 86)),
    Label("double other", 1, 1, "lane_mark", 0, False, False, (219, 194, 86)),
    Label("double white", 2, 2, "lane_mark", 0, False, False, (145, 219, 86)),
    Label("double yellow", 3, 3, "lane_mark", 0, False, False, (86, 219, 127)),
    Label("road curb", 4, 4, "lane_mark", 0, False, False, (86, 211, 219)),
    Label("single other", 5, 5, "lane_mark", 0, False, False, (86, 111, 219)),
    Label("single white", 6, 6, "lane_mark", 0, False, False, (160, 86, 219)),
    Label("single yellow", 7, 7, "lane_mark", 0, False, False, (219, 86, 178)),
]


# 目标检测类别定义
# 0: pedestrian
# 1: rider
# 2: car
# 3: truck
# 4: bus
# 5: train
# 6: motorcycle
# 7: bicycle
# 8: traffic light
# 9: traffic sign


# 语义分割类别定义
# 0:  road
# 1:  sidewalk
# 2:  building
# 3:  wall
# 4:  fence
# 5:  pole
# 6:  traffic light
# 7:  traffic sign
# 8:  vegetation
# 9:  terrain
# 10: sky
# 11: person
# 12: rider
# 13: car
# 14: truck
# 15: bus
# 16: train
# 17: motorcycle
# 18: bicycle


# 人姿态参数标签定义
# 0:  head
# 1:  neck
# 2:  right_shoulder
# 3:  right_elbow
# 4:  right_wrist
# 5:  left_shoulder
# 6:  left_elbow
# 7:  left_wrist
# 8:  right_hip
# 9:  right_knee
# 10: right_ankle
# 11: left_hip
# 12: left_knee
# 13: left_ankle
# 14: right_hand
# 15: left_hand
# 16: right_foot
# 17: left_foot
