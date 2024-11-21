import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix


# 使用 matplotlib 绘制混淆矩阵
def plot_confusion_matrix(
    y_true, y_pred, classes, title="Confusion Matrix", cmap=plt.cm.Blues, save_path=None
):
    """
    使用 matplotlib 绘制混淆矩阵。

    Parameters:
    - y_true: 真实标签（1D array）
    - y_pred: 预测标签（1D array）
    - classes: 类别名称
    - title: 图表标题
    - cmap: 热图的颜色
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    fig, ax = plt.subplots(figsize=(8, 6))

    cax = ax.matshow(cm_normalized, cmap=cmap)  # 绘制矩阵的颜色图
    fig.colorbar(cax)

    # 设置标签和标题
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # ax.set_title(title)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # 调整横轴标签的显示角度，防止重叠
    plt.xticks(rotation=45, ha="right", fontsize=8)  # 横轴标签旋转45度，并设置字体大小
    plt.yticks(fontsize=8)  # 设置纵轴标签的字体大小，与横轴标签一致

    # 在每个单元格中添加数字
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
                fontsize=8
            )

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )  # bbox_inches='tight' 防止保存时裁剪图像
        print(f"图像已保存至 {save_path}")
    plt.show()



def get_videolabel(videoid: str) -> int:
    """
    根据视频名称从 `video_annotations.json` 文件中提取对应的视频标签。

    该函数会读取存储在 `video_annotations.json` 文件中的 JSON 数据，并根据输入的视频名称 (`videoid`)
    查找对应的视频项。返回找到的视频的标签 (`label`)，如果没有找到相应的视频项，则返回 `None`。

    参数:
    videoid (str): 要查找的视频名称。该参数应为一个字符串，表示视频的标识符。

    返回:
    int: 如果找到对应的视频名称，返回视频的标签，类型为整数。如果未找到，返回 `None`。

    异常:
    如果 `video_annotations.json` 文件无法读取，可能会抛出 `FileNotFoundError` 或 `json.JSONDecodeError`。
    """

    # json数据存储在video_annotations.json文件中
    videoanno_path = r".\config\video_annotations.json"

    # 读取JSON文件
    try:
        with open(videoanno_path, "r", encoding="utf-8") as file:
            vs_json = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{videoanno_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{videoanno_path}' could not be parsed as JSON.")
        return None

    # 遍历列表，查找对应的video_name并提取label
    for video in vs_json:
        if video["videoName"] == videoid:
            label = video["label"]
            print(f"The label for video_id '{videoid}' is: {label}")
            return int(label)  # 返回整数类型的标签
    # 如果没有找到对应的视频名称
    else:
        print(f"Video_id '{videoid}' not found!!!!!!!!")
        return None


def interpolate_frames(matrix, target_frames):
    """
    对输入的17x2矩阵进行帧插值，将帧数从 X 调整到指定的 target_frames。

    参数：
    - matrix: 一个形状为(X, 17, 2)的numpy数组，表示 X 帧的姿态点数据，17个点的 xy 坐标。
    - target_frames: 目标帧数，范围为150~350。

    返回：
    - 插值后的矩阵，形状为(target_frames, 17, 2)。
    """
    # 判断矩阵是否合法
    if matrix.shape[0] <= 4:
        return matrix

    # 原始帧数
    original_frames = matrix.shape[0]

    # 创建一个帧的序列
    original_frame_indices = np.linspace(0, original_frames - 1, original_frames)
    target_frame_indices = np.linspace(0, original_frames - 1, target_frames)

    # 初始化插值结果
    interpolated_matrix = np.zeros((target_frames, matrix.shape[1], matrix.shape[2]))

    # 对每个姿态点（17个）进行插值
    for i in range(matrix.shape[1]):  # 17个姿态点
        for j in range(matrix.shape[2]):  # 每个姿态点的xy坐标
            # 创建插值函数
            interp_func = interp1d(
                original_frame_indices,
                matrix[:, i, j],
                kind="cubic",
                fill_value="extrapolate",
            )
            # 对该姿态点进行插值
            interpolated_matrix[:, i, j] = interp_func(target_frame_indices)

    return interpolated_matrix


def xy_normal(posarr):
    """
    将输入的数组进行归一化处理，使得所有坐标的 x 和 y 值都缩放到 [0, 1] 范围内。（对 x 和 y 单独 归一化）

    参数：
    posarr (numpy.ndarray): 一个形状为 (n, m, 2) 的三维数组，其中：
        - n 表示样本的帧数量（150~350）。
        - m 表示样本的姿态点数量（通常为 17）。
        - 2 表示每个坐标是一个二维点，包含 x 和 y 两个值。

    返回：
    numpy.ndarray: 一个与输入数组形状相同的三维数组，归一化后的坐标。形状为 (n, m, 2)，其中所有的 x 和 y 坐标均已被归一化至 [0, 1] 范围内。

    示例：
    >>> posarr = np.array([[[2, 3]], [[3, 1]], [[5, 6]]])
    >>> xy_normal(posarr)
    array([[[0.2, 0.3]],
           [[0.3, 0.1]],
           [[0.5, 0.6]]])
    """

    # 创建一个与输入数组相同形状的副本，以避免修改原数组
    resarr = np.copy(posarr)

    # 计算x坐标（第0维）的最小值和最大值
    x_min, x_max = np.min(resarr[:, :, 0]), np.max(resarr[:, :, 0])

    # 计算y坐标（第1维）的最小值和最大值
    y_min, y_max = np.min(resarr[:, :, 1]), np.max(resarr[:, :, 1])

    # 对x坐标进行归一化处理
    # 归一化公式： (x - x_min) / (x_max - x_min)
    resarr[:, :, 0] = (resarr[:, :, 0] - x_min) / (x_max - x_min)

    # 对y坐标进行归一化处理
    # 归一化公式： (y - y_min) / (y_max - y_min)
    resarr[:, :, 1] = (resarr[:, :, 1] - y_min) / (y_max - y_min)

    # 返回归一化后的坐标数组
    return resarr
