# 直观展示一个 npy 数组，shape = [2, x, 17]
# x 为帧数，一秒播放十帧， 2*17为十七个点的二维坐标

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设 keypoints 是已加载的 numpy 数组，形状为 [2, x, 17]
# 这里我们用随机数据代替，仅作演示用

fig, ax = plt.subplots()
(points,) = plt.plot([], [], "bo")  # 初始化一个空的点集


def init():
    ax.set_xlim(0, 1)  # 设置X轴范围，根据实际数据调整
    ax.set_ylim(0, 1)  # 设置Y轴范围，根据实际数据调整
    return (points,)


def update(frame, keypoints):
    # keypoints[0, frame, :] 表示 x 坐标，keypoints[1, frame, :] 表示 y 坐标
    points.set_data(keypoints[0, frame, :], keypoints[1, frame, :])
    return (points,)


if __name__ == "main":
    keypoints = np.load("../data/动作0-3-29.npy")
    ani = FuncAnimation(
        fig, update, frames=250, init_func=init, blit=True, interval=100
    )  # 100ms per frame

    plt.show()
