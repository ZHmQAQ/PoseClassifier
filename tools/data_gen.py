import os
import json
import numpy as np
import re
from src.datapro import PreProcess
from src.rtmpose_tran import RTM_Pose_Tran


def get_label_from_name(name):
    match = re.match(r"(\d+)-\d+\.mp4", name)
    if match:
        return match.group(1)
    return None


def process_video_file(video_path, video_labels, output_dir, file_to_label_mapping, use_filename_for_label=False):
    video_file = os.path.basename(video_path)
    label = get_label_from_name(video_file) if use_filename_for_label else video_labels.get(video_file)
    if label is not None:
        good_vid, keypoints = RTM_Pose_Tran(video_path)
        if keypoints.shape[0] >= 100:
            keypoints = PreProcess(keypoints)
        else:
            print("Data insufficient, discarding.")
            return
        output_filename = os.path.join(output_dir, video_file.replace(".mp4", ".npy"))
        np.save(output_filename, {"keypoints": keypoints, "label": label})
        file_to_label_mapping[output_filename] = label


def process_videos(directory, video_labels, output_dir, file_to_label_mapping, use_filename_for_label=False):
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path) and path.endswith(".mp4"):
            process_video_file(path, video_labels, output_dir, file_to_label_mapping, use_filename_for_label)


if __name__ == "__main__":
    filename = r"..\dataset\video_annotations.json"
    video_dir = r"..\dataset\TestData"
    output_dir = r"..\data\testdata"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    video_labels = {item["videoName"]: item["label"] for item in data}
    file_to_label_mapping = {}
    process_videos(video_dir, video_labels, output_dir, file_to_label_mapping, use_filename_for_label=True)
    with open(os.path.join(output_dir, "test_label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(file_to_label_mapping, f, ensure_ascii=False, indent=4)
