import cv2
import numpy as np
from rtmlib import Body
import torch

# 初始化Body模型
body = Body(
    pose="rtmo",  # 选择RTMO模型，它专注于身体姿态估计
    to_openpose=False,  # True为OpenPose风格，False为MMPose风格
    mode="lightweight",  # 可以选择 'balanced', 'performance', 'lightweight' 来调整性能和速度
    backend="onnxruntime",  # opencv, onnxruntime, openvino
    device="cuda",  # cpu, cuda, mps
)



def RTM_Pose_Tran(vid_path):
    """
    接受一个视频路径，输出关键点
    :param vid_path:
    :return:
    """
    # start = time.time()
    print(f"vid path = {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    result = []
    frame_num = 0

    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧
        if not ret:
            break  # 如果没有帧了，退出循环
        if frame_num % 2 == 0:  # 仅抽取偶数帧
            keypoints, _ = body(img)
            if len(keypoints) > 0:
                result.append(keypoints[0])
            else:
                print(f"no keypoints in this frame: {frame_num}")
        frame_num += 1
    cap.release()

    # end = time.time()

    return np.asarray(result)


if __name__ == "__main__":
    _ = RTM_Pose_Tran("test.mp4")
