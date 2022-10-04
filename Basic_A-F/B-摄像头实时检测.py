import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
"""
同济子豪兄：https://www.bilibili.com/video/BV1dL4y1h7Q6

"""

"""导入模型"""
# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
# 导入模型
pose = mp_pose.Pose(static_image_mode=False,         # 静态图片 or 连续帧视频
                    model_complexity=2,             # 人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于二者之间
                    smooth_landmarks=True,          # 平滑关键点
                    enable_segmentation=True,       # 人体抠图
                    min_detection_confidence=0.5,   # 置信度阈值
                    min_tracking_confidence=0.5)    # 各帧之间的追踪阈值

"""处理单帧的函数"""
def process_frame(img):
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型来预测结果
    results = pose.process(img_RGB)
    """可视化检测结果"""
    # 将关键点、连接和原图一同输出
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)    # 明明有这个属性，这里却提示找不到，但是可以正常运行，不管他

    return img


"""调用摄像头，逐帧实时处理每帧   （此函数为模板函数，任何应用只需修改单帧处理函数即可）"""
# 获取系统默认摄像头，0表示windows，1表示mac
cap = cv2.VideoCapture(0)
# 打开cap
cap.open(0)
# 无线循环，直到break 被触发
while cap.isOpened():
    # 获取页面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    # 处理单帧的函数
    frame = process_frame(frame)
    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:    # 按下q或esc退出（英文输入法下）
        break
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()