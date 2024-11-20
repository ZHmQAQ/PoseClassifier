import numpy as np
# from convert import convert
import os, shutil

import json


def convert(arr_y):
    arr = np.empty([200,17,2])
    # print(data.shape[0])
    for i, info in enumerate(arr_y):
        # info = info.T
        arr[i] = info
    new_arr = arr.transpose(2,0,1) # new_arr 是转换后的数据
    return new_arr


pattern = 1

# 假设json数据存储在standard.json文件中
filename = r'.\datapro\video_annotations.json'

# 读取JSON文件
with open(filename, 'r', encoding="utf-8") as file:
    data = json.load(file)

'.npy的地址'
folder = r'.\datapro\wuqin'
'''保存标准的数据集'''
save_folder = r'.\datapro\save'
save_label = '9'

arry = []

for file in os.listdir(folder):

    # 根据'_keypoints'分割文件名
    parts = file.split("_keypoints")
    # 获取'_keypoints'前面的部分
    video_id = parts[0]

    label = []
    # 遍历列表，查找对应的video_id并提取label
    for video in data:
        if video["videoId"] == video_id:
            label = video["label"]
            print(f"The label for video_id '{video_id}' is: {label}")
            break
    else:
        print(f"Video_id '{video_id}' not found.")
        

    file_s = os.path.join(folder, file)
    yuan_arr = np.load(file_s)

    # save_path = os.path.join(save_folder, '{:02d}'.format(int(label)), video_id)
    # np.save(save_path, yuan_arr)

    if pattern == 0:

        save_path = os.path.join(save_folder, '2_{:02d}'.format(int(label)), video_id)
        chou2_arr = [arr for i, arr in enumerate(yuan_arr) if (i+1) % 2 == 0]
        np.save(save_path, chou2_arr)
    
    elif pattern == 1:

        '''丢帧与补零，并且将同类组合为一个'''
        if label != save_label:
            continue

        new_arr = [arr for i, arr in enumerate(yuan_arr) if (i+1) % 4 == 0]

        if len(new_arr) > 200:
            final_arr = new_arr[0:200]

        elif len(new_arr) < 200:
            lenth = len(new_arr)
            for i in range(0,200-lenth):
                new_arr.append(np.zeros([17,2]))
            final_arr = new_arr

        else: 
            final_arr = new_arr
        
        final_arr = convert(final_arr)
        
        arry.append(final_arr)


save_path = os.path.join(save_folder, save_label)
np.save(save_path, arry)

