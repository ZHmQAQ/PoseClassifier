import numpy as np
import json
import os

from utils import xy_normal, interpolate_frames


FPS = 15  # 帧率
PRE_T = 1000  # 判定区间 Previous ms
NEX_T = 300   # 判定区间 Next     ms

PRE_FN = (FPS * PRE_T) // 1000  # 区间前面的帧数 frame number
NEX_FN = (FPS * NEX_T) // 1000  # 区间后面的帧数 frame number

ACTION_WEIGHT = [0.1, 0.25, 0.1, 0.25, 0.1, 0.05, 0.1, 0.05]

ERROR_LEVEL =  [[10, 20, 30],
                [10, 30, 50],
                [10, 20, 30],
                [10, 30, 50],
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30]]

ERROR_SCORE = [1.0, 0.8, 0.6, 0.0]

STD_FOLDER = r".\config\score_stddata"

# 判定模型超参数
ANGLE_LEVEL_RATIO = 2.86
POSES_ERROR_RATIO = 0.01
VEC_POS_ALPHA = 0.001


# 加载标准姿态数组
stdfiles = ["standard_00.npy", "standard_01.npy", "standard_02.npy", "standard_03.npy", 
            "standard_04.npy", "standard_05.npy", "standard_06.npy", "standard_07.npy",
            "standard_08.npy", "standard_09.npy", "standard_10.npy", "standard_11.npy",
            "standard_12.npy", "standard_13.npy"]
std_normposes_arr = {}
for STD_file_i in os.listdir(STD_FOLDER):
    if STD_file_i in stdfiles:
        idx = stdfiles.index(STD_file_i)
        fp = os.path.join(STD_FOLDER, STD_file_i)
        tmparr = np.load(fp)
        std_normposes_arr[idx] = tmparr


# 由两个向量计算角度
def vec_angle(vec1, vec2):
    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算向量的模
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_a * norm_b)

    # 计算夹角（以弧度为单位）
    return np.degrees(np.arccos(cos_theta))


# 由姿态矩阵计算角度
def cpt_angle(posarr):
    # 维度 17 * 2
    angle1 = vec_angle(posarr[ 8] - posarr[ 6], posarr[12] - posarr[ 6])
    angle2 = vec_angle(posarr[10] - posarr[ 8], posarr[ 6] - posarr[ 8])
    angle3 = vec_angle(posarr[11] - posarr[ 5], posarr[ 7] - posarr[ 5])
    angle4 = vec_angle(posarr[ 5] - posarr[ 7], posarr[ 9] - posarr[ 7])
    angle5 = vec_angle(posarr[ 6] - posarr[12], posarr[14] - posarr[12])
    angle6 = vec_angle(posarr[ 5] - posarr[11], posarr[13] - posarr[11])
    angle7 = vec_angle(posarr[12] - posarr[14], posarr[16] - posarr[14])
    angle8 = vec_angle(posarr[11] - posarr[13], posarr[15] - posarr[13])
    return np.array((angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8))


# 定义关键动作角度比较
def cpr_angle(angles, angles_std):
    scores = []
    allerror = np.abs(angles - angles_std) * ANGLE_LEVEL_RATIO
    for i in range(allerror.shape[0]):
        ind = np.digitize(allerror[i], ERROR_LEVEL[i])
        scores.append(ERROR_SCORE[ind])
    return np.sum(np.array(scores) * ACTION_WEIGHT)


# 矩阵维度规范化
def arr_std(resarr, len):
    # resarr = np.copy(posarr)  # 不拷贝，加快点速度
    # 获取矩阵的帧数
    frames = resarr.shape[0]

    # 如果帧数不足，则添加最后一帧
    if frames < len:
        # 获取最后一帧
        last_frame = resarr[-1]
        # 计算需要添加的帧数
        additional_frames = len - frames
        # 添加帧
        resarr = np.concatenate((resarr, last_frame[np.newaxis, ...].repeat(additional_frames, axis=0)), axis=0)
    # 如果帧数多于，则丢掉多余的帧
    elif frames > len:
        # 保留前50帧
        resarr = resarr[:len]
    return resarr


def cpt_ferror(frame):
    # 计算一帧的误差，即矩阵内17*2所有元素绝对值求和
    return np.sum(np.abs(frame)) / frame.size


def sliding_window_error(error_matrix, window_size):
    # 计算滑动窗口的总误差
    num_frames = error_matrix.shape[0]
    num_windows = num_frames - window_size + 1
    total_errors = np.zeros(num_frames)
    
    for i in range(num_windows):
        # 选取窗口内的数据
        window = error_matrix[i:i+window_size]
        # 计算窗口内所有帧的误差总和
        total_errors[i+PRE_FN] = np.min([cpt_ferror(frame) for frame in window])
    
    total_errors[0:PRE_FN]  = total_errors[PRE_FN]
    total_errors[-NEX_FN:] = total_errors[-NEX_FN-1]
    return total_errors


def cpt_posdiff(normposes):
    # 计算前一帧与后一帧的差值，构成一个插值矩阵49*17*2
    diff_matrix = np.diff(normposes, axis=0)

    # 在前面补充第一帧全零
    padded_diff_matrix = np.pad(diff_matrix, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
    
    return padded_diff_matrix


def score_arr(arr, label, data, std_normposes_arr):

    
    # 动作误差计算  std_normposes 与 normposes
    std_normposes = std_normposes_arr[label]
    # normposes = arr_std(xy_normal(arr), std_normposes.shape[0])
    arr = interpolate_frames(arr, std_normposes.shape[0])
    normposes = xy_normal(arr)

    std_normposes_diff = cpt_posdiff(std_normposes)
    normposes_diff = cpt_posdiff(normposes)

    # error_poses = np.abs(std_normposes - normposes)
    error_poses = (1 - VEC_POS_ALPHA) * np.abs(std_normposes_diff - normposes_diff) + \
        VEC_POS_ALPHA * np.abs(std_normposes - normposes)

    window_error_poses = sliding_window_error(error_poses, NEX_FN + PRE_FN + 1)

    # 姿态角度
    std_poses = data[label]['allPosesAng']
    keyframe = [FPS * x for x in data[label]['allFrameSec']]
    key_num = len(keyframe)
    keyi = 0  # 关键帧计数器

    keyscore = np.zeros(key_num)  # 关键动作得分
    AddOne_flag = False

    for currf in range(arr.shape[0]):

        if keyi < key_num and (currf >= (keyframe[keyi] - NEX_FN) and currf <= (keyframe[keyi] + PRE_FN)):
            AddOne_flag = True
            if window_error_poses[currf] > POSES_ERROR_RATIO:
                continue
            tmpangle = cpt_angle(arr[currf])
            tmpscore = cpr_angle(tmpangle, std_poses[keyi])
            
            keyscore[keyi] = max(keyscore[keyi], tmpscore)
        else:
            if AddOne_flag:
                keyi += 1  # 计数
                AddOne_flag = False

    totalscore = np.sum(keyscore) / key_num
    return totalscore


def Score(npy, label):

    # 加载JSON文件
    with open(os.path.join(STD_FOLDER, 'parsed_data.json'), 'r') as file:
        data = json.load(file)

    # arr = np.load(npy)

    return score_arr(npy, label, data, std_normposes_arr)


if __name__ == "__main__":

    # video_path = os.path.join(STD_FOLDER, "test_10.npy")
    video_path = os.path.join(r".\datapro\save\2_08","动作8-5-85.npy")
    for i in range(14):
        res = Score(npy=video_path, label=i)
        print("label ", i, " - score:", res)
    # res = score_npy(npy=video_path, label=0)
    # print("score:", res)
