## <div align="center">ScriptTool模块说明</div>

ScriptTool模块为各项通用脚本工具,包括重命名、数据复制、名称删选等基本脚本工具。

<details open>
<summary><b><font color=Indigo>目录文件</font></b></summary>
<font color=CornflowerBlue>CA目录</font>完成基于<b>目标检测</b>任务的相关数据格式转换,具体参考<a href='./CA/README.md'>说明</a>。
</details>

<p></p>

<details open>
<summary><b><font color=Indigo>脚本文件</font></b></summary>
<font color=CornflowerBlue>BatchRename.py</font>完成批量文件的重命名。

<font color=CornflowerBlue>ObjectImageWeightSample.py</font>针对序列标注的目标检测图像,根据图像中出现的框的个数进行随机采样,采样过程可能会出现重复目标采样。
<font color=CornflowerBlue>SplitTrainValDataset.py.py</font>当前数据的标注情况进行数据集的训练集和验证集的分离,从而保证可靠性,该采样过程不会出现重复目标采样。
</details>