## <div align="center">Visualization模块说明</div>

Visualization模块为满足直观的数据统计、观测和比较等进行可视化操作，便于整体数据处理的Bug处理和数据挖掘、分析、特征提取等。

<font color=CornflowerBlue>semantics_color.py</font>针对类别PNG图基本是纯黑的情况，进行着色的颜色表示，便于了解实际的标注情况。
<font color=CornflowerBlue>object_bbox.py</font>针对标签输出为YOLO的cxywh的情况进行图像绘制检验，确保后续检查，也为其他潜在的调用提供接口。<b>Arial.ttf</b>是用于Pillow库优化显示所用。
<font color=CornflowerBlue>BDD100K_label.py</font>针对BDD数据集的相关标签的定义和可视化进行了说明。