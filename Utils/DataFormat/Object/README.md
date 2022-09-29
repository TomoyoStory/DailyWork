# <div align="center">Object模块说明</div>

Object模块完成各类数据针对<b>目标检测</b>任务的数据格式转换。

<details open>
<summary><b><font color=Indigo>脚本文件</font></b></summary>

<font color=CornflowerBlue>BDD2YOLO.py</font> 完成<b>BDD数据集</b>向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换,内含关于单独的交通灯数据格式提取  
<font color=CornflowerBlue>BSTLD2YOLO.py</font> 完成<b>Bosch Small Traffic Lights Dataset</b>的yaml文件数据格式向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  
<font color=CornflowerBlue>CATrafficLight2YOLO.py</font> 完成<b>CA数据集</b>中的交通灯相关数据向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  
<font color=CornflowerBlue>COCO2YOLO.py</font> 完成<b>COCO数据集</b>向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  
<font color=CornflowerBlue>LabelMe2YOLO.py</font> 完成<b>LabelMe标注工具</b>得到的数据格式向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  
<font color=CornflowerBlue>TT100k2YOLO.py</font> 完成<b>TTK数据集</b>向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  
<font color=CornflowerBlue>VOC2YOLO.py</font> 完成<b>VOC格式</b>的xml文件向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换  

</details>

<p></p>

<details open>
<summary><b><font color=Indigo>目录文件</font></b></summary>

<font color=CornflowerBlue>Caltech Pedestrian Detection目录</font>包括<b>Caltech Pedestrian Detection行人数据集</b>的.seq文件和.vbb向<b>YOLOv5</b>数据格式(c,x,y,w,h)的数据格式的转换所需的工具脚本。
</details>