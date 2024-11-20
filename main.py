import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.data_feeder import TrainFeeder, InferFeeder
from src.model import ST_GCN


print("Use CUDA:", torch.cuda.is_available())
print("torch version:", torch.__version__)


seed = 123
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


def my_train_with_cross_validation():
    from sklearn.model_selection import KFold
    from src.early_stopping import EarlyStopping  # 自己实现的早停实现

    NUM_EPOCH = 2000  # 因为加入了 early stop 所以这个可以设置高一点。
    PATIENCE = 200  # early stop 的 patience 参数
    BATCH_SIZE = 128
    best_acc = 0
    K_FOLDS = 5
    datapro = None  # 可设置为 None / combined transform 注意不要括号，因为是传函数对象
    model_path = Path("model")
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    best_model_path = model_path / "best_model.pth"

    # 加载数据集
    data = np.load("data/train_data.npy")
    labels = np.load("data/train_label.npy")
    # print(f"data shape: {data.shape}")
    # print(f"labels shape: {labels.shape}")

    # # debug: 检查 label 是否越界
    # num_classes = 14  # 你的类别总数
    # assert labels.max() < num_classes, "Label index exceeds number of classes"
    # assert labels.min() >= 0, "Label index is negative"

    kf = KFold(n_splits=K_FOLDS, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        # 实例化模型
        # 交叉验证每折都要重置模型，不然等于是在之前的基础上继续训练，必会 100% 准确率
        model = ST_GCN(
            num_classes=14,
            in_channels=2,  # 二维
            t_kernel_size=9,  # 时间图卷积的内核大小(t_kernel_size × 1)
            hop_size=1,
        ).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
            patience=PATIENCE, verbose=True, path=str(model_path / f"fold_{fold}_best_model.pth")
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
