import numpy as np
import random
import src.utils

# 数据增强
def random_translation(data, max_translation=10):
    # 随机平移
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            max_translation = random.choice([2, 20, 200])
            t = np.random.uniform(-max_translation, max_translation)
            data[i, j, :, :] += t
    return data


def random_rotation(data, max_angle=20):
    # 随机旋转
    # 二维平面旋转
    angle = np.random.uniform(-max_angle, max_angle)
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
        ]
    )
    # 三维绕轴旋转（三个旋转矩阵）
    # rotation_matrix = np.array([
    #         [1, 0, 0],
    #         [0, np.cos(angle), -np.sin(angle)],
    #         [0, np.sin(angle), np.cos(angle)]
    #     ])
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            # tmp = data[i, :, j, :]
            # print(data[i, :, j, :].shape)
            # print(data[i, :, j, :])
            data[i, :, j, :] = np.dot(rotation_matrix, data[i, :, j, :])
    return data


def random_scaling(data, max_scale=2):
    # 随机缩放
    for i in range(data.shape[0]):
        scale = np.random.uniform(-max_scale, max_scale)
        # tmp = (2 ** scale)
        data[i, :, :, :] *= 2**scale
    return data


def random_sampling(data, min_ratio=0.7, max_ratio=0.9):
    """
    从输入的 Y * 2 * X * 17 数据矩阵中随机采样一个连续的帧片段。
    :param data: 输入的数据矩阵，形状为 (Y, 2, X, 17)
    :param min_ratio: 最小采样比率，默认为 0.7
    :param max_ratio: 最大采样比率，默认为 0.9
    :return: 采样后的数据矩阵，形状为 (Y, 2, sample_length, 17)
    """
    Y, _, X, _ = data.shape
    # 采样的结果存储
    sampled_data = []

    for i in range(Y):
        # 计算采样长度，随机选择一个在 [min_ratio * X, max_ratio * X] 范围内的值
        sample_length = random.randint(int(min_ratio * X), int(max_ratio * X))
        # 随机选择一个起始帧，保证采样不超出原始帧数
        start_frame = random.randint(0, X - sample_length + 1)
        # 提取该区间的数据
        sampled_data_i = data[i, :, start_frame : start_frame + sample_length, :]
        sampled_data_i = src.utils.interpolate_frames(sampled_data_i.transpose(1, 2, 0), X).transpose(2, 0, 1)
        sampled_data.append(sampled_data_i)

    return np.asarray(sampled_data)


def combined_transform(data):
    # 随机相机运动
    data = random_rotation(data)
    data = random_scaling(data)
    data = random_translation(data)
    # 随机采样片段
    data = random_sampling(data)
    return data


def PreProcess(result):
    """
    1. 帧数插值
    2. 轴变换
    将 X * 17 * 2 插值到 Y * 17 * 2 （Y=220），然后轴变换 transpose(2, 0, 1) 到 2 * X * 17
    """

    new_arr = src.utils.interpolate_frames(result, 250)
    final_arr = new_arr.transpose(2, 0, 1)

    return final_arr


# 用于读取数据的函数
# class Feeder(torch.utils.data.Dataset):
#
#     def __init__(self, data_path, label_path, transform=False):
#         super().__init__()
#         label = np.load(label_path)
#         data = np.load(data_path)
#         # 数据增强的变换
#         if transform:
#             transform = lambda x: random_scaling(random_rotation(random_translation(x)))
#             augmented_data = transform(np.copy(data))
#             label = np.concatenate((label, np.copy(label)), axis=0)
#             data = np.concatenate((data, augmented_data), axis=0)
#
#         self.label = label
#         self.data = data
#
#     def __len__(self):
#         return len(self.label)
#
#     def __iter__(self):
#         return self
#
#     def __getitem__(self, index):
#         data = np.array(self.data[index])
#         label = self.label[index]
#         return data, label
