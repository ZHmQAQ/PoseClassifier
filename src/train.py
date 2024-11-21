import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

from src.data_feeder import TrainFeeder, InferFeeder
from src.model import ST_GCN
from datapro import combined_transform
from utils import plot_confusion_matrix


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


def train():
    from sklearn.model_selection import StratifiedKFold
    from src.early_stopping import EarlyStopping  # 早停

    NUM_EPOCH = 2000  # 因为加入了 early stop 所以这个可以设置高一点无所谓
    PATIENCE = 30  # early stop 的 patience 参数
    BATCH_SIZE = 128
    best_acc = 0
    K_FOLDS = 5
    datapro = None  # 可设置为 None / combined transform 注意不要括号，因为是传函数对象
    model_path = Path("../model")
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    best_model_path = model_path / "best_model_2.pth"

    # 加载数据集
    data = np.load("../tools/train_keypoints.npy")
    labels = np.load("../tools/train_labels.npy")
    # print(f"data shape: {data.shape}")
    # print(f"labels shape: {labels.shape}")

    # # debug: 检查 label 是否越界
    # num_classes = 14  # 你的类别总数
    # assert labels.max() < num_classes, "Label index exceeds number of classes"
    # assert labels.min() >= 0, "Label index is negative"

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        # 实例化模型
        # 交叉验证每折都要重置模型，不然等于是在之前的基础上继续训练，必会 100% 准确率
        model = ST_GCN(
            num_classes=15,
            in_channels=2,  # 二维
            t_kernel_size=9,  # 时间图卷积的内核大小(t_kernel_size × 1)
            hop_size=1,
        ).cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,  # 学习率
            weight_decay=1e-4,  # L2 正则化系数（权重衰减）
        )
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training fold {fold+1}/{K_FOLDS}")
        # # debug: 检查 index 是否越界
        # print(f"len(data):{len(data)}")
        # assert train_index.max() < len(
        #     data
        # ), f"Training index = {train_index.max()} is out of bounds."
        # assert val_index.max() < len(
        #     data
        # ), f"Validation index = {val_index.max()} is out of bounds."

        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # 使用 np.unique 统计每个元素的出现次数
        unique_elements, counts = np.unique(val_labels, return_counts=True)
        # 输出结果
        for element, count in zip(unique_elements, counts):
            print(f"元素 {element} 出现了 {count} 次")

        # 创建 dataloader
        # 注意这里要用 train_feeder 类是因为交叉验证需要直接传入 npy 而不是路径
        # 注意格式转换，要转换成 float 和 long
        train_dataset = TrainFeeder(train_data, train_labels, transform=datapro)
        val_dataset = TrainFeeder(val_data, val_labels, transform=datapro)

        data_loader = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            ),
            "val": torch.utils.data.DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False
            ),
        }

        # 创建早停实例
        early_stopping = EarlyStopping(
            patience=PATIENCE,
            verbose=True,
            path=str(model_path / f"fold_{fold}_best_model.pth"),
        )

        # 训练模型
        for epoch in range(1, NUM_EPOCH + 1):
            model.train()
            total_loss = 0
            correct = 0
            for __data, label in data_loader["train"]:
                __data, label = __data.cuda(), label.cuda()
                optimizer.zero_grad()
                output = model(__data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()

            # 计算训练集上的准确率
            train_acc = 100 * correct / len(data_loader["train"].dataset)

            # 在验证集上进行验证并计算损失
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for _data, label in data_loader["val"]:
                    _data, label = _data.cuda(), label.cuda()
                    output = model(_data)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == label).sum().item()

            val_loss /= len(data_loader["val"].dataset)
            val_acc = 100 * correct / len(data_loader["val"].dataset)
            print(
                f"Fold {fold+1}, Epoch {epoch}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # 调用早停
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at fold {fold+1}, epoch {epoch}")
                break

            # 更新并保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("Best Validation Accuracy across folds: {:.2f}%".format(best_acc))


def train_2():
    from sklearn.model_selection import KFold
    from src.early_stopping import EarlyStopping  # 早停

    NUM_EPOCH = 230  # 因为加入了 early stop 所以这个可以设置高一点无所谓
    PATIENCE = 30  # early stop 的 patience 参数
    BATCH_SIZE = 128
    best_tracc, best_vaacc, best_valoss, best_epoch = 0, 0, 10, -1

    datapro = combined_transform  # 可设置为 None / combined transform 注意不要括号，因为是传函数对象
    model_path = Path("../model")
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    best_model_path = model_path / "best_model_2.pth"

    # 加载数据集
    data = np.load("../tools/train_keypoints.npy")
    labels = np.load("../tools/train_labels.npy")
    # print(f"data shape: {data.shape}")
    # print(f"labels shape: {labels.shape}")

    # # debug: 检查 label 是否越界
    # num_classes = 14  # 你的类别总数
    # assert labels.max() < num_classes, "Label index exceeds number of classes"
    # assert labels.min() >= 0, "Label index is negative"

    # 实例化模型
    # 交叉验证每折都要重置模型，不然等于是在之前的基础上继续训练，必会 100% 准确率
    model = ST_GCN(
        num_classes=15,
        in_channels=2,  # 二维
        t_kernel_size=9,  # 时间图卷积的内核大小(t_kernel_size × 1)
        hop_size=1,
    ).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,  # 学习率
        weight_decay=2e-4,  # L2 正则化系数（权重衰减）
    )
    criterion = torch.nn.CrossEntropyLoss()

    from sklearn.model_selection import train_test_split

    # 设置划分比例，80% 用于训练集，20% 用于测试集
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.1, random_state=42
    )
    # 打印数据集的大小
    print(f"Training data size: {train_data.shape}, Test data size: {val_data.shape}")
    # train_data, val_data = data[0:670], data[670:]
    # train_labels, val_labels = labels[0:670], labels[670:]

    # 使用 np.unique 统计每个元素的出现次数
    unique_elements, counts = np.unique(val_labels, return_counts=True)
    # 输出结果
    for element, count in zip(unique_elements, counts):
        print(f"元素 {element} 出现了 {count} 次")

    # 创建 dataloader
    # 注意这里要用 train_feeder 类是因为交叉验证需要直接传入 npy 而不是路径
    # 注意格式转换，要转换成 float 和 long
    train_dataset = TrainFeeder(train_data, train_labels, transform=datapro)
    val_dataset = TrainFeeder(val_data, val_labels, transform=datapro)

    data_loader = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        ),
    }

    # 创建早停实例
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        verbose=True,
        path=str(model_path / f"earlystop_best_model.pth"),
    )

    # 训练模型
    for epoch in range(1, NUM_EPOCH + 1):
        model.train()
        total_loss = 0
        correct = 0
        for __data, label in data_loader["train"]:
            __data, label = __data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(__data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()

        # 计算训练集上的准确率
        train_acc = 100 * correct / len(data_loader["train"].dataset)

        # 在验证集上进行验证并计算损失
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for _data, label in data_loader["val"]:
                _data, label = _data.cuda(), label.cuda()
                output = model(_data)
                loss = criterion(output, label)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()

        val_loss /= len(data_loader["val"].dataset)
        val_acc = 100 * correct / len(data_loader["val"].dataset)
        print(
            f"Epoch {epoch}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # 调用早停
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print(f"epoch {epoch}")
        #     break

        # 更新并保存最佳模型
        if val_acc > best_vaacc:
            best_vaacc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        elif val_acc == best_vaacc and train_acc > best_tracc:
            best_tracc = train_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        elif (
            val_acc == best_vaacc and train_acc == best_tracc and val_loss < best_valoss
        ):
            best_valoss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    model = ST_GCN(num_classes=15, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(best_model_path))
    classes = (
        [f"八段锦{i}" for i in range(8)] + [f"五禽戏{i}" for i in range(6)] + ["其他"]
    )
    # 训练完成后，绘制训练集和验证集的混淆矩阵
    print("\nConfusion Matrix on Training Set:")
    evaluate_and_plot_confusion_matrix(model, data_loader["train"], classes, save_path='..\model\TRA_confusion_matrix.png')

    print("\nConfusion Matrix on Validation Set:")
    evaluate_and_plot_confusion_matrix(model, data_loader["val"], classes, save_path='..\model\VAL_confusion_matrix.png')

    print("best Epoch:", best_epoch, " : ", best_vaacc, best_tracc, best_valoss)


def test_model():
    """
    验证训练后的模型
    """
    pass


if __name__ == "__main__":
    train_2()
