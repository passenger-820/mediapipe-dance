import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
"""
同济子豪兄：https://www.bilibili.com/video/BV1dL4y1h7Q6

"""

# 定义可视化图像函数
def look_img(img):
    """opencv读取图像格式为RGB，因此需要将BGR转RGB"""
    # 解决Pycharm中使用cv2代码无法自动补全的问题: https://blog.csdn.net/Z2572862506/article/details/125243384
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

"""
导入模型
"""
# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
# 导入模型,第一次运行mediapipe 代码的时候 有时候会下载模型，但是有时候因为网络问题，可能下载不下来，报错.
# 参考博客解决办法 https://blog.csdn.net/m0_57110410/article/details/125538796
pose = mp_pose.Pose(static_image_mode=True,         # 静态图片 or 连续帧视频
                    model_complexity=2,             # 人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于二者之间
                    smooth_landmarks=True,          # 平滑关键点
                    enable_segmentation=True,       # 人体抠图
                    min_detection_confidence=0.5,   # 置信度阈值
                    min_tracking_confidence=0.5)    # 各帧之间的追踪阈值

"""
读入图像
"""
# 从图片文件读入图像，opencv读入的是RGB格式
img = cv2.imread('zihao.png')
look_img(img)

"""
将图像输入模型，获取预测结果
"""
# BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型来预测结果
results = pose.process(img_RGB)

"""
可视化检测结果
"""
# 将关键点、连接和原图一同输出
mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)    # 明明有这个属性，这里却提示找不到，但是可以正常运行，不管他
look_img(img)
# 也可以在三维真实物理坐标系中可视化
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

