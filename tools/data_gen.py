import os
import json
import numpy as np
from src.datapro import PreProcess
from src.rtmpose_tran import RTM_Pose_Tran

# 读取 JSON 文件
filename = r"..\dataset\video_annotations.json"
video_dir = r"..\dataset"
output_dir = r"..\data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(filename, "r", encoding="utf-8") as file:
    data = json.load(file)
# 创建一个字典，以视频名称为键，标签为值
video_labels = {item["videoName"]: item["label"] for item in data}

# 创建一个字典来存储文件名到标签的映射，用于保存到 JSON 文件中
file_to_label_mapping = {}


def process_videos(directory):
    # 遍历给定目录中的所有文件和子目录
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            # 如果是目录，假设是子目录，继续遍历其中的文件
            process_videos(path)
        elif path.endswith(".mp4"):
            # 如果是文件且为视频文件，处理视频
            process_video_file(path)


def process_video_file(video_path):
    video_file = os.path.basename(video_path)
    if video_file in video_labels:
        # 使用 RTM_Pose_Tran 处理视频文件，获取关键点
        _, keypoints = RTM_Pose_Tran(video_path)
        keypoints = PreProcess(keypoints)
        print(f"shape: {keypoints.shape}")
        # 根据视频文件名从字典中获取对应的标签
        label = video_labels[video_file]
        # 构建输出文件名
        output_filename = os.path.join(output_dir, video_file.replace(".mp4", ".npy"))
        # 保存关键点数据和标签
        np.save(output_filename, {"keypoints": keypoints, "label": label})
        # 更新文件到标签的映射字典
        file_to_label_mapping[output_filename] = label


if __name__ == "__main__":
    process_videos(video_dir)
    # 将文件到标签的映射保存为 JSON 文件
    with open(
        os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(file_to_label_mapping, f, ensure_ascii=False, indent=4)
