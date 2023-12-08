# <div align="center">文档说明</div>

<p>
这个日常工作脚本仓库的目的在于对日常工作中的基于AI的相关数据、深度学习模块架构、论文笔记参考等综合内容的统计接口开发和维护,从而实现代码的可复用性。该代码当前处于日常开发的过程中,还未进行第一版v1.0的tag发布。潜在问题、需求和BUG请联系本人。
</p>

<details open>
<summary><b><font color=Crimson>使用需求</font></b></summary>

[**Python>=3.6.0**](https://www.python.org/) 版本需要
[requirements.txt](./requirements.txt) 相关包需求
```bash
$ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev 
  (上面部分为docker环境部署时候欠缺对应库所需)
$ git clone https://github.com/TomoyoStory/DailyWork
$ cd DailyWork
$ pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
  (国内使用阿里源)
```
</details>

<details open>
<summary><b><font color=CadetBlue >各个目录描述</font></b></summary>
<font color=CornflowerBlue>Utils</font>目录为各种类别工具,详情参考<a href='./Utils/README.md'>说明</a>。

<font color=CornflowerBlue>LinuxShell</font>目录为各种Linux操作系统脚本工具,详情参考<a href='./LinuxShell/README.md'>说明</a>。
</details>


# <div align="center"><font color=Crimson>Coming Soon</font></div>
其他内容当前还未完成,敬请期待。