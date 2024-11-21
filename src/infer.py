import torch
from src.data_feeder import TrainFeeder, InferFeeder
from src.model import ST_GCN


def infer():
    """
    加载模型 best_model.pth
    加载视频 转化为关键点坐标图 (暂时略过，直接读取关键点坐标 npy 数据)
    模型处理坐标图 输出分类
    """

    model_path = r"model/bestbest/best_model_2.pth"
    data_path = r"dataset/TestData.npy"

    # Load the model
    model = ST_GCN(num_classes=15, in_channels=2, t_kernel_size=9, hop_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data loader
    dataset = InferFeeder(data_path, "npy")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference
    results = []  # 记录推理结果以便返回
    for data in data_loader:
        data = data.float()  # Ensure data is in float
        action, conf = model.predict(data.numpy())  # 调用 predict 方法
        results.append([action, conf])

    print(results)
    return results