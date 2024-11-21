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

# 姿态点之间的连接关系
neighbor_base = [
    (0, 1),
    (1, 2),
    (2, 0),
    (2, 4),
    (1, 3),
    (6, 4),
    (5, 3),
    (8, 6),
    (5, 7),
    (6, 5),
    (8, 10),
    (7, 9),
    (12, 6),
    (14, 12),
    (16, 14),
    (11, 5),
    (11, 13),
    (13, 15),
]


# 绘制姿态点
def draw_pose_and_connections(frame, keypoints):
    # 绘制连接线
    for start, end in neighbor_base:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        # 绘制线条（连接两个姿态点）
        cv2.line(
            frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )  # 绿色，线宽2

    for point in keypoints:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绿色圆圈，半径5

    return frame


def RTM_Pose_Tran(vid_path, display_pose=False):
    """
    接受一个视频路径，输出关键点
    :param vid_path:
    :return:
    """

    print(f"vid path = {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames = {total_frames}")
    # print(f"Total frames: {total_frames}")\
    RATIO = 0.5
    INTERVAL = 3  # 抽帧间隔，每 x 帧抽一帧
    SCALE = 600  # 窗口显示尺寸（高）
    result = []
    good_vid = True  # 判断视频有人类的帧数是否超过 FRAME_THRESHOLD
    frame_num = 0
    frame_count = 0

    if display_pose:
        # 定义展示的窗口尺寸
        window_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        window_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # 设置 OpenCV 窗口的大小
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pose Estimation", int(SCALE * window_width / window_height), SCALE)

    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧
        if not ret:
            break  # 视频结束，退出循环

        if frame_num % INTERVAL == 0:
            # 获取姿态点
            keypoints, _ = body(img)
            if len(keypoints) > 0:
                result.append(keypoints[0])
                frame_count += 1

                # 可视化姿态点
                if display_pose:
                    cv2.imshow(
                        "Pose Estimation", draw_pose_and_connections(img, keypoints[0])
                    )
                    # 按键 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        frame_num += 1  # 增加帧计数器
    cap.release()
    cv2.destroyAllWindows()

    if frame_count * INTERVAL < total_frames * RATIO:
        good_vid = False

    return good_vid, np.asarray(result)


if __name__ == "__main__":
    _ = RTM_Pose_Tran("test.mp4")
