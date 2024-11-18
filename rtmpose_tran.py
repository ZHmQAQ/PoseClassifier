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
    cap = cv2.VideoCapture(vid_path)
    result = []
    frame_num = 0

    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧
        if not ret:
            break  # 如果没有帧了，退出循环
        if frame_num % 2 == 0:  # 仅抽取偶数帧
            keypoints, _ = body(img)
            result.append(keypoints[0])
        frame_num += 1
    cap.release()

    # end = time.time()
    new_arr = [arr for i, arr in enumerate(result) if (i + 1) % 2 == 0]

    print(len(new_arr))

    if len(new_arr) > 200:
        final_arr = new_arr[0:200]

    elif len(new_arr) < 200:
        length = len(new_arr)
        for i in range(0, 200 - length):
            new_arr.append(np.zeros([17, 2]))
        final_arr = new_arr

    else:
        final_arr = new_arr

    final_arr = np.asarray(final_arr)
    final_arr = final_arr.transpose(2, 0, 1)

    print(final_arr.shape)
    return final_arr


if __name__ == "__main__":
    _ = RTM_Pose_Tran("test.mp4")
