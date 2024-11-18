# 分为 train 和 val，有不同的功能。

import numpy as np
import torch
from torch.utils.data import Dataset
from rtmpose_tran import RTM_Pose_Tran


class TrainFeeder(Dataset):
    def __init__(self, data, labels, transform=None):
        super(TrainFeeder, self).__init__()
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

        # 如果有转换函数且需要数据增强
        if self.transform:
            # 应用数据增强
            augmented_data = self.transform(np.copy(data))
            # 格式转换成 f32
            augmented_data = augmented_data.astype(np.float32)
            self.data = np.concatenate((self.data, augmented_data), axis=0)
            self.labels = np.concatenate((self.labels, np.copy(self.labels)), axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label


class ValFeeder(Dataset):
    def __init__(self, data_path, label_path=None, data_type="npy"):
        super(ValFeeder, self).__init__()
        if data_type == "npy":
            self.data = np.load(data_path)
            self.labels = np.load(label_path) if label_path else None
        elif data_type == "vid":
            self.data = RTM_Pose_Tran(data_path)
            self.labels = np.load(label_path) if label_path else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index] if self.labels is not None else None
        # Return data and label if available, or just data if no labels
        return (data, label) if label is not None else data
