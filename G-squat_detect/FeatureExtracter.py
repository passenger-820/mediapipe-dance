from matplotlib import pyplot as plt
import os
import csv
import numpy as np
from BootstrapHelper import BootstrapHelper
from BodyEmbedder import FullBodyPoseEmbedder
from PoseClassifier import PoseClassifier

# 指定训练集路径
bootstrap_images_in_folder = 'squat_dataset'

# Output folders for bootstrap images and CSVs
bootstrap_images_out_folder = 'squat_images_out'
bootstrap_csvs_out_folder = 'squat_csvs_out'

# 初始化helper: 对训练集中的图片进行特征提取
bootstrap_helper = BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=bootstrap_csvs_out_folder,
)

# 检查每个动作有多少张图像
bootstrap_helper.print_images_in_statistics()

# 提取特征
bootstrap_helper.bootstrap(per_pose_class_limit=None)

# 检查每个动作有多少张图像提取了特征
bootstrap_helper.print_images_out_statistics()

# After iinitial bootstrapping images without deteced poses were still saved in
# the folder (but not in the CSVs) for debug purpose. Let's remove them.
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()

""""
检查异常样本
"""
# Align CSVs with filtered images.
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()

# Transforms pose landmarks into embedding.
pose_embedder = FullBodyPoseEmbedder()

# Classifies give pose against database of poses.
pose_classifier = PoseClassifier(
    pose_samples_folder=bootstrap_csvs_out_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

outliers = pose_classifier.find_pose_sample_outliers()
print('Number of outliers: ', len(outliers))

# 查看所有异常数据点
bootstrap_helper.analyze_outliers(outliers)

# 移除异常数据点
bootstrap_helper.remove_outliers(outliers)

# 重新整理二分类数据
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()


"""
特征生成CSV文件
"""


def dump_for_the_app():
    pose_samples_folder = 'squat_csvs_out'
    pose_samples_csv_path = 'squat_csvs_out_basic.csv'
    file_extension = 'csv'
    file_separator = ','

    # Each file in the floder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    with open(pose_samples_csv_path, 'w') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-len(file_extension) + 1]

            # One file line: 'sample_00001,x1,y1,x2,y2,...'
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)

    # files.download(pose_samples_csv_path) # 这应该是针对谷歌云盘下载到本地的


dump_for_the_app()
