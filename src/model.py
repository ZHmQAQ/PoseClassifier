import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
