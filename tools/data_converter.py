"""
对 data_gen 之后的 npy 数据做进一步处理
"""

import os
import numpy as np


# input_npy = r"..\data\standard_9.npy"
input_dir = r"..\data\testdata"

keypointsnpy = []
labelsnpy = []

for file in os.listdir(input_dir):
    filepath = os.path.join(input_dir, file)
    arr = np.load(filepath, allow_pickle=True).item()

    keypointsnpy.append(arr["keypoints"])
    labelsnpy.append(int(arr["label"]))


print(len(keypointsnpy))
print(len(labelsnpy))
np.save("../data/test_keypoints/test_keypoints.npy", np.asarray(keypointsnpy))
np.save("../data/test_keypoints/test_labels.npy", np.asarray(labelsnpy))
