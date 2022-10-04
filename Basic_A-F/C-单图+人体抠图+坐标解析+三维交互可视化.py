import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

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
# 导入模型
pose = mp_pose.Pose(static_image_mode=True,  # 静态图片 or 连续帧视频
                    model_complexity=2,  # 人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于二者之间
                    smooth_landmarks=True,  # 平滑关键点
                    enable_segmentation=True,  # 人体抠图
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)  # 各帧之间的追踪阈值

"""
读入图像，输入模型，获取预测结果
"""
# 从图片文件读入图像，opencv读入的是RGB格式
img = cv2.imread('zihao.png')
# BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型来预测结果
results = pose.process(img_RGB)

"""
人体抠图结果
"""
mask = results.segmentation_mask # mask表示每一个像素对应人体的概率
mask = mask > 0.5
plt.imshow(mask)
plt.show()
# 单通道转3通道
mask_3 = np.stack((mask, mask, mask), axis=-1)
MASK_COLOR = [0, 200, 0]
fg_image = np.zeros(img.shape, dtype=np.uint8)
fg_image[:] = MASK_COLOR
# 获得前景人像
FG_img = np.where(mask_3, img, fg_image)
look_img(FG_img)
# 获得抠掉前景人像的背景
BG_img = np.where(~mask_3, img, fg_image)
look_img(BG_img)

"""
所有关键点检测结果
"""
# print(results.pose_landmarks)
# print(mp_pose.POSE_CONNECTIONS)
# # 左胳膊肘的归一化坐标
# print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW])  # 名称
# print(results.pose_landmarks.landmark[13])   # 代号
# print(results.pose_landmarks.landmark[13].x)   # x坐标

"""
解析指定关键点的像素坐标
"""
# # 想得到关键点在原图上的位置，y要乘上原图高度，x要乘上原图宽度
# h = img.shape[0]
# w = img.shape[1]
# print(results.pose_landmarks.landmark[13].x * w)    # 左胳膊肘关键点像素横坐标
# print(results.pose_landmarks.landmark[13].y * h)    # 左胳膊肘关键点像素纵坐标

"""
解析指定关键点的真实物理（米）坐标
"""
# print(results.pose_world_landmarks[mp_pose.PoseLandmark.NOSE])
# # 真实物理坐标的远点位于左右髋关节连线的中点（肚脐附近），详见论文
# print(results.pose_world_landmarks[23])

"""
交互式三维可视化
"""
# coords = np.array(results.pose_landmarks.landmark)  # 诡异的数据结构转为规整的np.array类型
# print(len(coords))
# print(coords)
# print(coords[0].x)
#
#
# def get_x(each):
#     return each.x
#
#
# def get_y(each):
#     return each.y
#
#
# def get_z(each):
#     return each.z
#
#
# # 分别获取所有关键点的XYZ坐标
# # map 可以对列表里的每个元素，逐一的用函数操作!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# points_x = np.array(list(map(get_x, coords)))
# points_y = np.array(list(map(get_y, coords)))
# points_z = np.array(list(map(get_z, coords)))
# # 将3个方向的坐标合并
# points = np.vstack((points_x, points_y, points_z)).T
# print(points)  # 33行分别代表33个关键点，3列分别代表XYZ
#
# import open3d as o3d
#
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([point_cloud])
