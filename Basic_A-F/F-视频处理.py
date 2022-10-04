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


"""视频逐帧处理模板   （此函数为模板函数，任何应用只需修改单帧处理函数即可）"""
def generate_vedio(input_path='videos/your.file',file_type='mp4'):
    file_head = input_path.split('/')[-1]
    output_path = 'videos_processed/out-' + file_head

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while(cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('your name')
    cap = cv2.VideoCapture(input_path)
    # 获取视频窗口宽、高
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    """
    # fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数，注意：字符顺序不能弄混
    # cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
    # cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    """
    if file_type=='mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 子豪兄是mp4的
    if file_type=='flv':
        fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')

    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                try:
                    process_frame(frame)
                except:
                    print('error')
                    pass

                if success:
                    out.write(frame)
                    # 进度条更新一帧
                    pbar.update(1)
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

# 些一次只能运行一个，否则连续跑多个，后面的会没效果
# generate_vedio(input_path='videos/585726284-1-80.flv',file_type='flv')
# generate_vedio(input_path='videos/556758755-1-80.flv',file_type='flv')
# generate_vedio(input_path='videos/267843079_nb2-1-80.flv',file_type='flv')
# generate_vedio(input_path='videos/sport1.mp4',file_type='mp4')
# generate_vedio(input_path='videos/sport2.mp4',file_type='mp4')
# generate_vedio(input_path='videos/sport3.mp4',file_type='mp4')
# generate_vedio(input_path='videos/mycountry.flv',file_type='flv')
