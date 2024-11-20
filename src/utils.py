import numpy as np
from scipy.interpolate import interp1d



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
            interp_func = interp1d(original_frame_indices, matrix[:, i, j], kind='cubic', fill_value="extrapolate")
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
