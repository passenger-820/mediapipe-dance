import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

"""
同济子豪兄：https://www.bilibili.com/video/BV1dL4y1h7Q6

"""


# 定义可视化图像函数
def look_img(img):
    """opencv读取图像格式为RGB，因此需要将BGR转RGB"""
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
# 导入模型
pose = mp_pose.Pose(static_image_mode=True,  # 静态图片 or 连续帧视频
                    model_complexity=2,  # 人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于二者之间
                    smooth_landmarks=True,  # 平滑关键点
                    enable_segmentation=True,  # 人体抠图
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)  # 各帧之间的追踪阈值

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
mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 明明有这个属性，这里却提示找不到，但是可以正常运行，不管他
look_img(img)

"""
获取左膝盖关键点像素坐标
"""
h = img.shape[0]
w = img.shape[1]
# # 非空，则证明检测出来了
# print(results.pose_landmarks)
# # 左膝盖关键点像素坐标
# cx = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w)
# cy = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
# cz = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z)
# print(cx, cy, cz)
# # 绘制图：图像，圆心坐标，半径，RGB颜色，最后一个参数为线宽，-1表示填充
# img = cv2.circle(img, (cx, cy), 15, (255, 0, 0), -1)
# look_img(img)

# 可视化关键点及骨架连线
mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

for i in range(33):  # 遍历33个关键点
    # 获取关键点的三维坐标
    cx = int(results.pose_landmarks.landmark[i].x * w)
    cy = int(results.pose_landmarks.landmark[i].y * h)
    cz = int(results.pose_landmarks.landmark[i].z)

    radius = 3

    if i == 0:  # 鼻尖
        img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
    elif i in [11, 12]:     # 肩膀
        img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
    elif i in [23, 24]:     # 髋关节
        img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
    elif i in [13, 14]:     # 胳膊肘
        img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
    elif i in [25, 26]:     # 膝盖
        img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
    elif i in [15, 16, 27, 28]:     # 手腕和脚腕
        img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
    elif i in [17, 19, 21]:     # 左手
        img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
    elif i in [18, 20, 22]:     # 右手
        img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
    elif i in [27, 29, 31]:     # 左脚
        img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
    elif i in [28, 30, 32]:     # 右脚
        img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
    elif i in [9, 10]:      # 嘴
        img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
    elif i in [1, 2, 3, 4, 5, 6, 7, 8]:     # 眼及脸颊
        img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
    else:     # 其他关键点
        img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)

look_img(img)
