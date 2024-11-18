import torch
import numpy as np
import random

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
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
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
        data[i, :, :, :] *= (2 ** scale) 
    return data


def combined_transform(data):
    data = random_rotation(data)
    data = random_scaling(data)
    data = random_translation(data)
    return data




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


