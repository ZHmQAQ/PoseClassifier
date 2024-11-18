import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_feeder import TrainFeeder, InferFeeder
from datapro import combined_transform

print("Use CUDA:", torch.cuda.is_available())
print("torch version:", torch.__version__)


# 用于读取数据的函数
class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path=None):
        super().__init__()
        self.data = np.load(data_path).astype(np.float32)
        # 只有在提供了标签路径时才加载标签
        if label_path is not None:
            self.labels = np.load(label_path).astype(np.int64)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.labels is not None:
            label = self.labels[index]
            return data, label
        else:
            return data


class Graph:
    def __init__(self, hop_size):
        # 规定边缘排列，作为集合{{起点,终点},{起点,终点},{起点,终点……}这样规定一个边为元素。
        self.get_edge()

        # hop: hop数连接几个分离的关节
        # 例如hop=2的话，手腕不仅和胳膊肘连在一起，还和肩膀连在一起。
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, hop_size=hop_size
        )

        # 创建一个相邻矩阵。在这里，根据hop数创建一个相邻矩阵。
        # hop是2的时候，0hop, 1hop, 2hop这三个相邻的矩阵被创建。
        # 论文中提出了多种生成方法。这次使用了简单易懂的方法。
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 17
        self_link = [(i, i) for i in range(self.num_node)]  # Loop
        neighbor_base = [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 4),
            (1, 3),
            (6, 4),
            (5, 3),
            (8, 6),
            (5, 7),
            (6, 5),
            (8, 10),
            (7, 9),
            (12, 6),
            (14, 12),
            (16, 14),
            (11, 5),
            (11, 13),
            (13, 15),
        ]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))  # 邻接矩阵
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(A, Dn)
        return DAD


class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * s_kernel_size,
            kernel_size=1,
        )

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # 对邻接矩阵进行GC，相加特征。
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous()


class STGC_block(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5
    ):
        super().__init__()
        self.sgc = SpatialGraphConvolution(
            in_channels=in_channels, out_channels=out_channels, s_kernel_size=A_size[0]
        )

        # Learnable weight matrix M 给边缘赋予权重。学习哪个边是重要的。
        self.M = nn.Parameter(torch.ones(A_size))

        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel_size, 1),  # kernel_size
                (stride, 1),  # stride
                ((t_kernel_size - 1) // 2, 0),  # padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x


class ST_GCN(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])  # 75

        # STGC_blocks
        self.stgc1 = STGC_block(
            in_channels, 32, 1, t_kernel_size, A_size
        )  # in_c=3, t_k_s= 9， 1是步长
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)

        # Prediction
        self.fc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        # print("ST-GCN input:",x.shape) # ST-GCN input: torch.Size([128, 3, 80, 25])

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        # print("ST-GCN input reshape 之后:",x.shape) # ST-GCN input reshape 之后: torch.Size([128, 75, 80])
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # print("ST-GCN input做完一维BN后形状：",x.shape) # ST-GCN input做完一维BN后形状： torch.Size([128, 3, 80, 25])
        # 给我的感觉是对25个关键点的xyz做了个归一化
        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)

        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x

    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients to speed up this part
            x = torch.tensor(
                x, dtype=torch.float32
            )  # Ensure input tensor is of correct type
            if x.dim() == 3:  # If single sample, add batch dimension
                x = x.unsqueeze(0)
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)  # Convert output to probabilities
            top_p, top_class = probabilities.topk(
                1, dim=1
            )  # Find the highest probability class
        return (
            top_class.numpy(),
            top_p.numpy(),
        )  # Return class and the corresponding probability


seed = 123
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


def my_train():
    NUM_EPOCH = 300
    BATCH_SIZE = 128
    acc_list = []
    best_acc = 0
    best_model_path = "best_model.pth"  # 最好模型的保存路径
    final_model_path = "final_model.pth"  # 训练结束后最终模型的保存路径

    # 实例化模型
    model = ST_GCN(
        num_classes=4,
        in_channels=2,  # 二维
        t_kernel_size=9,  # 时间图卷积的内核大小(t_kernel_size × 1)
        hop_size=1,
    ).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 数据集的准备
    data_loader = dict()
    data_loader["train"] = torch.utils.data.DataLoader(
        dataset=Feeder(
            data_path="data/train_data.npy", label_path="data/train_label.npy"
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    data_loader["test"] = torch.utils.data.DataLoader(
        dataset=Feeder(
            data_path="data/test_data.npy", label_path="data/test_label.npy"
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 将模型转变为学习模式
    model.train()

    # 开始训练
    for epoch in range(1, NUM_EPOCH + 1):
        correct = 0
        sum_loss = 0
        for batch_idx, (data, label) in enumerate(data_loader["train"]):
            data = data.cuda()
            label = label.cuda()

            output = model(data)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()

        acc = 100.0 * correct / len(data_loader["train"].dataset)
        acc_list.append(acc)

        # 更新并保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)

        print(
            "# Epoch: {} | Loss: {:.4f} | Accuracy: {:.4f} | best acc: {:.4f}".format(
                epoch, sum_loss / len(data_loader["train"].dataset), acc, best_acc
            )
        )

    print("Best Accuracy: {}".format(best_acc))

    # 保存训练结束后的最终模型
    torch.save(model.state_dict(), final_model_path)


def my_train_with_cross_validation():
    from sklearn.model_selection import KFold
    from early_stopping import EarlyStopping  # 自己实现的早停实现

    NUM_EPOCH = 2000  # 因为加入了 early stop 所以这个可以设置高一点。
    BATCH_SIZE = 128
    best_acc = 0
    K_FOLDS = 5
    best_model_path = "best_model.pth"  # 最好模型的保存路径

    # 加载数据集
    data = np.load("data/train_data.npy")
    labels = np.load("data/train_label.npy")
    print(f"data shape: {data.shape}")
    print(f"labels shape: {labels.shape}")

    # debug: 检查 label 是否越界
    num_classes = 14  # 你的类别总数
    assert labels.max() < num_classes, "Label index exceeds number of classes"
    assert labels.min() >= 0, "Label index is negative"

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
        # debug: 检查 index 是否越界
        print(f"len(data):{len(data)}")
        assert train_index.max() < len(
            data
        ), f"Training index = {train_index.max()} is out of bounds."
        assert val_index.max() < len(
            data
        ), f"Validation index = {val_index.max()} is out of bounds."

        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # 创建 dataloader
        # 注意这里要用 train_feeder 类是因为交叉验证需要直接传入 npy 而不是路径
        # 注意格式转换，要转换成 float 和 long
        train_dataset = TrainFeeder(
            train_data, train_labels, transform=combined_transform
        )
        val_dataset = TrainFeeder(
            val_data, val_labels, transform=combined_transform
        )

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
            patience=100, verbose=True, path=f"fold_{fold}_best_model.pth"
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

    model_path = "best_model.pth"
    data_path = r"data\val_data.npy"

    # Load the model
    model = ST_GCN(num_classes=14, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data loader
    dataset = InferFeeder(data_path, 'npy')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference
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


def recognize_actions_and_scores_in_video(model, data_path, data_type='npy'):
    # 加载和预处理视频数据
    import time

    dataset = Feeder(data_path, data_type)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    start_time = time.time()
    for keypoints in data_loader:
        keypoints = keypoints.float()  # 确保数据类型正确
        labels, scores = model.predict(keypoints)  # 使用模型进行预测
    end_time = time.time()

    cost_time = (end_time - start_time) * 1000  # 转换时间为毫秒

    actions_scores_times = [
        {"label": labels.tolist(), "score": scores.tolist(), "times": cost_time},
    ]

    print(actions_scores_times)
    return actions_scores_times


if __name__ == "__main__":
    # my_train_with_cross_validation()
    my_inference()
