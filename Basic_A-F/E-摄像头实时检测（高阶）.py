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
    # 记录开始处理的时间
    start_time = time.time()
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型来预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks:  # 若检测出人体关键点
        # 可视化关键点、骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)    # 明明有这个属性，这里却提示找不到，但是可以正常运行，不管他

        for i in range(33):  # 遍历33个关键点
            # 获取关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = int(results.pose_landmarks.landmark[i].z)

            radius = 3

            if i == 0:  # 鼻尖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11, 12]:  # 肩膀
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:  # 髋关节
                img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:  # 胳膊肘
                img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:  # 膝盖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:  # 手腕和脚腕
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
            elif i in [17, 19, 21]:  # 左手
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            elif i in [18, 20, 22]:  # 右手
                img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
            elif i in [27, 29, 31]:  # 左脚
                img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
            elif i in [28, 30, 32]:  # 右脚
                img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [9, 10]:  # 嘴
                img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 眼及脸颊
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            else:  # 其他关键点
                img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)

    else:
        scaler = 1
        failure_str = "No Person"
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 0), 2 * scaler)

    # 记录该帧处理完毕时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)

    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 0), 2 * scaler)


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
    # 处理单帧的函数!!!!!!!!!!!!!!!!
    process_frame(frame)
    # frame = process_frame(frame) # 不需要这样，毕竟没有返回值
    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:    # 按下q或esc退出（英文输入法下）
        break
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()