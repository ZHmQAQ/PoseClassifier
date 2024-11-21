import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import plot_confusion_matrix


# 获取模型的预测结果并计算混淆矩阵
def evaluate_and_plot_confusion_matrix(model, data_loader, classes, save_path=None):
    """
    在给定的数据集上评估模型并绘制混淆矩阵。

    Parameters:
    - model: 训练好的模型
    - data_loader: 数据加载器（可以是训练集或验证集）
    - classes: 类别名称（例如 ['0', '1', '2', ..., '9']）
    - save_path: 图片保存
    """
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, predicted = output.max(1)
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 绘制混淆矩阵
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统中
    plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示为乱码
    plot_confusion_matrix(np.array(all_labels), np.array(all_preds), classes, save_path=save_path)


def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证过程中的损失（Loss）曲线。

    Parameters:
    - train_losses: 训练集的损失值列表
    - val_losses: 验证集的损失值列表
    - save_path: 如果提供，损失曲线会保存为图片。默认为 None，表示不保存。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_losses, label="Training Loss", color="blue", linestyle="-", linewidth=2
    )
    plt.plot(
        val_losses, label="Validation Loss", color="red", linestyle="--", linewidth=2
    )
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 保存图像（可选）
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")

    plt.show()
