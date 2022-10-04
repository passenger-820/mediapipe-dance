# zihaoDance

我自己笔记本是py3.8，实验室电脑是py3.9，所以手动下载安装mediapipe切勿装错版本。

笔记本电脑Pycharm中使用cv2代码无法自动补全，解决办法参考链接https://blog.csdn.net/Z2572862506/article/details/125243384

导入模型,第一次运行mediapipe 代码的时候 有时候会下载模型，但是有时候因为网络问题，可能下载不下来，报错：远程链接拒绝。这是因为现在是从谷歌网盘下，国内你懂的。参考博客解决办法 https://blog.csdn.net/m0_57110410/article/details/125538796

# 构建分类器
## 训练集数据结构
俩个文件夹，分别存放深蹲的两个极端情况--站着和完全蹲下。
而且都是周围360°拍摄的照片，完整的人，不存在遮挡

* squat_dataset
    * up
        * image_001.jpg 
        * image_002.jpg 
        ...
    * down
        * image_001.jpg 
        * image_002.jpg 
        ...

# 其他说明
可视化模块的字体部分，由于源代码要连外网下，所以灭活了一部分，如有需要可后续自行修改

# package
python               >= 3.8
opencv-python        4.5.1.48
mediapipe            0.8.11
numpy                1.22.1
tqdm                 4.64.1

