"""
对 data_gen 之后的 npy 数据做进一步处理
"""

import os
import numpy as np


input_npy = r"..\data\standard_9.npy"
input_dir = r"..\data"

train_keypointsnpy = []
train_labelsnpy = []

for file in os.listdir(input_dir):
    filepath = os.path.join(input_dir, file)
    arr = np.load(filepath, allow_pickle=True).item()

    train_keypointsnpy.append(arr["keypoints"])
    train_labelsnpy.append(int(arr["label"]))


print(len(train_keypointsnpy))
print(len(train_labelsnpy))
np.save("train_keypoints.npy", np.asarray(train_keypointsnpy))
np.save("train_labels.npy", np.asarray(train_labelsnpy))
