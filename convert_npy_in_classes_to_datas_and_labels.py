# 现在有数个 k.npy 文件，0.npy ~ 3.npy 形状是 (50, 2, 200, 17)
# k 是动作序号或者说 label， 50 是视频个数，2 是关键点维度，17 是关键点个数， 200 是帧数。
# 需要合并这些文件，分成一组 train_data.npy, 一组 test_data.npy
# train_data.npy 形状是 (160, 2, 200, 17), test_data.npy 形状是 (40, 2, 200, 17), 也就是说 200 个不同种类的视频被分为训练集和测试集，并且每个 class 分出 40 个 train 10 个 test。
# 此外，生成数据集的时候，顺便生成相应的label， train_label.npy 和 test_label.npy ，记录相应的data来自于哪个标签。

import numpy as np
from pathlib import Path

# 假设你已经将数据文件放在了当前工作目录或指定路径下
base = Path("分别组合")
file_paths = [
    "0.npy",
    "1.npy",
    "2.npy",
    "3.npy",
    "4.npy",
    "5.npy",
    "6.npy",
    "7.npy",
    "8.npy",
    "9.npy",
    "10.npy",
    "11.npy",
    "12.npy",
    "13.npy",
]

# 读取所有文件
data = [np.load(base / path) for path in file_paths]


# 随机划分训练数据和测试数据
np.random.seed(42)  # 设置随机种子以保证可复现性
train_data = []
val_data = []
train_labels = []
val_labels = []

# 生成对应的标签
for i, cls_data in enumerate(data):
    # 随机打乱索引
    indices = np.random.permutation(len(cls_data))
    train_indices = indices[:40]  # 选取前40个作为训练数据
    val_indices = indices[40:50]  # 选取接下来10个作为验证数据

    train_data.append(cls_data[train_indices])
    val_data.append(cls_data[val_indices])

    train_labels.append(np.full(40, i))  # 生成训练标签
    val_labels.append(np.full(10, i))  # 生成验证标签


# 将列表转换为numpy数组
train_data = np.concatenate(train_data, axis=0)
val_data = np.concatenate(val_data, axis=0)
train_labels = np.concatenate(train_labels)
val_labels = np.concatenate(val_labels)

# 保存处理后的数据和标签
np.save("data/train_data.npy", train_data)
np.save("data/val_data.npy", val_data)
np.save("data/train_label.npy", train_labels)
np.save("data/val_label.npy", val_labels)

# 输出数据和标签的形状，以确保一切正常
print("Train data shape:", train_data.shape)
print("Test data shape:", val_data.shape)
print("Train label shape:", train_labels.shape)
print("Test label shape:", val_labels.shape)
