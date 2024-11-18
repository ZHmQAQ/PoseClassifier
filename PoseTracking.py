import os
import time
import argparse
import pandas as pd
import torch

from rtmpose_tran import RTM_Pose_Tran
from score import Score


# 加载预训练模型的函数
def load_model(model_path=r"model\best_model.pth"):
    from model import ST_GCN

    model = ST_GCN(num_classes=14, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 处理视频并识别动作分类及其标准度分数的函数
# 这里只需要处理单个视频
def recognize_actions_and_scores_in_video(model, video_path):
    start = time.time()
    # 视频转关键点
    keypoints = RTM_Pose_Tran(video_path)
    # 关键点输入模型，取得分类
    action, conf = 1, 1
            # (model.predict(keypoints))
    if conf < 0.5:
        action = 14  # 置信度过低，分类到“其它”
        score = 0
    else:
        # 关键点输入打分代码，取得评分
        score = Score(keypoints, action)
    end = time.time()
    duration = end - start
    return action, score, duration


# 将结果写入CSV文件的函数
def write_results_to_csv(results, output_csv):
    data = []
    for video_name, (action, score, duration) in results.items():
        data.append([video_name, action, score, duration])

    df = pd.DataFrame(data, columns=["视频名称", "动作分类", "标注度评分", "推理耗时"])
    df.to_csv(output_csv, index=False)


# 主函数
def main(args):
    video_directory = args.video_directory  # 视频文件目录
    output_csv = os.path.join(
        args.result_directory, f"{args.phone_number}_submit.csv"
    )  # 输出的CSV文件路径

    # 加载模型
    model = load_model()

    # 初始化一个字典来存储结果
    results = {}

    # 遍历视频目录中的文件
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            # 识别视频中的动作及其分数
            action, score, duration = recognize_actions_and_scores_in_video(
                model, video_path
            )
            # 存储识别结果
            results[filename] = (action, score, duration)

    # 将结果写入CSV文件
    write_results_to_csv(results, output_csv)


# 设置命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频动作识别和打分")

    # 设置默认值的例子
    parser.add_argument(
        "--video_directory",
        type=str,
        default="vid",  # 默认视频文件目录
        help="视频文件目录",
    )
    parser.add_argument(
        "--result_directory",
        type=str,
        default="res",  # 默认结果输出目录
        help="结果输出目录",
    )
    parser.add_argument(
        "--phone_number",
        type=str,
        default="1234567890",  # 默认的队长手机号
        help="队长手机号",
    )

    args = parser.parse_args()

    main(args)
