from matplotlib import pyplot as plt
import os
import csv
import numpy as np
import tqdm
import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from BodyEmbedder import FullBodyPoseEmbedder
from PoseClassifier import PoseClassifier
from Utils import EMADictSmoothing, show_image
from Counter import RepetitionCounter
from Visualizer import PoseClassificationVisualizer

"""
对视频进行预测
"""
video_path = 'squat_vedios/squat5.mp4'  # 需要进行预测的视频名称
class_name = 'down'     # 以阈值超过哪一个类别判断动作成功了
out_video_path = 'squat_vedios/squat5-output.mp4'   # 输出视频的文件名
bootstrap_csvs_out_folder = 'squat_csvs_out'


video_cap = cv2.VideoCapture(video_path)

# Get some video parameters to generate output video with classification
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize tracker, classifier and counter.
# Do that before every video as all of them have state.


# Folder with pose class CSVs. That should be the same folder you using while
# buliding classifier to output CSVs.
pose_samples_folder = 'squat_csvs_out'

# Initialize tracker
pose_tracker = mp_pose.Pose()
# pose_tracker = mp_pose.Pose(upper_body_onle=False) 这句应该不用中间有参数!!!!!!!!!!!!

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=bootstrap_csvs_out_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removeing them): ', len(outliers))

# Initialize EMA smoothing
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# 指定两个动作的阈值
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as 'top_n_by_mean_distance'
    plot_y_max=10)


# Open output video.
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                            video_fps, (int(video_width), int(video_height)))

frame_idx = 0
output_frame = None
with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while True:
        # Get next frame of the video.
        success, frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(image=input_frame)
        pose_landmarks = results.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)

            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classifier the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth classification use EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)

            # Count repetitions.
            repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to mataining correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetition count.
            repetitions_count = repetition_counter.n_repeats

        # Draw classification plot and repetition counter.
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # Save the output frame.
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        # Show intermediate frames of the video to track progress.
        if frame_idx % 50 == 0:
            show_image(output_frame)

        frame_idx += 1
        pbar.update()

    # Close output video.
    out_video.release()

    # Release mediapipe resources.
    pose_tracker.close()

    # Show the last frame of the video.
    if output_frame is not None:
        show_image(output_frame)


# 这应该是针对谷歌云盘下载到本地的: 下载生成的视频
# files.download(out_video_path)