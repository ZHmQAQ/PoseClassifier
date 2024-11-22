import numpy as np
from torch.utils.data import Dataset


class TrainFeeder(Dataset):
    def __init__(self, data, labels, transform=None, augment_factor=4):
        super(TrainFeeder, self).__init__()
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

        # 如果有转换函数且需要数据增强
        if self.transform:
            # 应用数据增强
            for i in range(augment_factor):
                augmented_data = self.transform(np.copy(data))
                # 格式转换成 f32
                augmented_data = augmented_data.astype(np.float32)
                self.data = np.concatenate((self.data, augmented_data), axis=0)
                self.labels = np.concatenate((self.labels, np.copy(labels)), axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label


class InferFeeder(Dataset):
    # 预期行为： as a label-less feeder
    # not correctly implemented
    def __init__(self, data_path, data_type="npy"):
        super(InferFeeder, self).__init__()
        if data_type == "npy":
            self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data
