import numpy as np
import torch

from pathlib import Path
from src.data_feeder import TrainFeeder
from src.model import ST_GCN
from src.visualize import evaluate_and_plot_confusion_matrix


def test_model(model_path, save_path):
    """
    验证训练后的模型
    """
    _model_path = model_path
    _save_path = save_path

    model = ST_GCN(num_classes=14, in_channels=2, t_kernel_size=9, hop_size=1)

    keypoints = np.load(r"../../data/test_keypoints/test_keypoints.npy")
    labels = np.load(r"../../data/test_keypoints/test_labels.npy")

    dataset = TrainFeeder(keypoints, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model.load_state_dict(torch.load(_model_path))
    classes = (
        [f"八段锦{i}" for i in range(8)] + [f"五禽戏{i}" for i in range(6)]
    )
    # 训练完成后，绘制训练集和验证集的混淆矩阵
    print("\nConfusion Matrix on Training Set:")
    evaluate_and_plot_confusion_matrix(
        model,
        data_loader,
        classes,
        save_path=_save_path,
    )

if __name__ == "__main__":
    base = Path(".")
    for model_file in base.glob("*.pth"):
        print(f"Found model file: {model_file}")
        save_path = model_file.parent / f"{model_file.stem}_TEST_confusion_matrix.png"
        print(f"Save path: {save_path}")
        test_model(str(model_file), str(save_path))
