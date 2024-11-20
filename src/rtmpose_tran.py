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

    print(f"vid path = {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    result = []
    FRAME_THRESHOLD = 180
    good_vid = True  # 判断视频有人类的帧数是否超过 FRAME_THRESHOLD
    frame_num = 0
    frame_count = 0

    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧
        if not ret:
            break  # 视频结束，退出循环

        if frame_num % 2 == 0:
            keypoints, _ = body(img)
            if len(keypoints) > 0:
                result.append(keypoints[0])
                frame_count += 1

        frame_num += 1  # 增加帧计数器
    cap.release()

    if frame_count < FRAME_THRESHOLD:
        good_vid = False

    return good_vid, np.asarray(result)




if __name__ == "__main__":
    _ = RTM_Pose_Tran("test.mp4")
