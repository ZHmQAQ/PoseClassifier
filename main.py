import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.data_feeder import TrainFeeder, InferFeeder
from src.model import ST_GCN
from src.train import train

print("Use CUDA:", torch.cuda.is_available())
print("torch version:", torch.__version__)


def my_train():
    train()


def my_inference():
    """
    加载模型 best_model.pth
    加载视频 转化为关键点坐标图 (暂时略过，直接读取关键点坐标 npy 数据)
    模型处理坐标图 输出分类
    """

    model_path = r"model\best_model.pth"
    data_path = r"data\val_data.npy"

    # Load the model
    model = ST_GCN(num_classes=14, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data loader
    dataset = InferFeeder(data_path, "npy")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference
    results = []  # 记录推理结果以便返回
    for data in data_loader:
        data = data.float()  # Ensure data is in float
        with torch.no_grad():
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            print("Predicted class:", predicted)
            print("Confidences:")
            for i in range(
                probabilities.shape[0]
            ):  # Loop over all samples in the batch
                print(f"Sample {i} predictions:")
                for idx, prob in enumerate(probabilities[i]):
                    print(f"  Class {idx}: {prob.item():.4f}")


def my_inference1():
    """
    加载模型 best_model.pth
    加载视频 转化为关键点坐标图 (暂时略过，直接读取关键点坐标 npy 数据)
    模型处理坐标图 输出分类
    调用模型内置的 predict 模块
    """
    model_path = r"model\best_model.pth"
    data_path = r"data\val_data.npy"

    # Load the model
    model = ST_GCN(num_classes=14, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data loader
    dataset = InferFeeder(data_path, "npy")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference using the built-in predict method
    results = []  # 记录推理结果以便返回
    for data in data_loader:
        data = data.float()  # Ensure data is in float
        top_class, top_p = model.predict(data.numpy())  # 调用 predict 方法

        # Log and store the results
        for idx, (cls, prob) in enumerate(zip(top_class, top_p)):
            print(f"Sample {idx} predictions:")
            print(f"  Class {cls[0]}: {prob[0]:.4f}")
            results.append({
                "predicted_class": cls[0],
                "probability": prob[0]
            })

    return results


if __name__ == "__main__":
    # my_train_with_cross_validation()
    my_inference1()
