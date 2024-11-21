# 基于人体姿态关键点的视频分类和打分模型

该项目整合了 **RTMPose 姿态识别**和 **ST-GCN 时空图卷积**两个模型，结合开创性的**距离角度判定算法**成功实现了“基于人体姿态关键点的视频分类和打分任务”。

# 任务说明和实现效果演示

以中国传统文化中的“八段锦”“五禽戏”共计 14 个动作为分类目标，使用“中国移动动感地带AI+高校创智计划”提供的数据集，在 14 个类别上实现高效推理和评分。

## 输出结果展示
在验证集上的输出如下表展示。可以看到，模型成功完成了对动作的分类和打分。

| 视频名称              | 动作分类 | 标注度评分 | 推理耗时  |
|-------------------|------|-------|-------|
| reference_10.mp4  | 10   | 0.52  | 7.03  |
| reference_157.mp4 | 14   | 0.00  | 13.33 |
| reference_158.mp4 | 14   | 0.00  | 13.10 |
| reference_159.mp4 | 14   | 0.00  | 13.07 |
| reference_160.mp4 | 14   | 0.00  | 12.96 |
| reference_161.mp4 | 14   | 0.00  | 13.08 |
| reference_8.mp4   | 8    | 0.61  | 8.66  |
| standard_10.mp4   | 10   | 1.00  | 6.48  |
| standard_8.mp4    | 8    | 1.00  | 8.32  |
| 动作0-4-29.mp4      | 0    | 0.86  | 6.46  |
| 动作0-4-43.mp4      | 0    | 0.76  | 6.53  |
| 动作0-4-44.mp4      | 0    | 0.64  | 6.50  |
| 动作0-5-1.mp4       | 0    | 0.78  | 6.91  |
| 动作1-4-3.mp4       | 1    | 0.45  | 6.37  |
| 动作1-4-30.mp4      | 1    | 0.57  | 5.85  |
| 动作1-4-4.mp4       | 1    | 0.60  | 6.32  |
| 动作1-4-45.mp4      | 1    | 0.47  | 6.03  |
| 动作10-1-13.mp4     | 10   | 0.52  | 6.57  |
| 动作10-1-2.mp4      | 10   | 0.54  | 6.45  |
| 动作10-1-8.mp4      | 10   | 0.51  | 6.39  |
| 动作10-3-3.mp4      | 10   | 0.56  | 6.29  |
| 动作11-10-14.mp4    | 11   | 0.49  | 5.99  |
| 动作11-10-28.mp4    | 8    | 0.50  | 6.30  |
| 动作11-10-42.mp4    | 11   | 0.50  | 6.10  |
| 动作11-10-55.mp4    | 11   | 0.47  | 5.80  |
| 动作12-12-45.mp4    | 12   | 0.82  | 5.56  |
| 动作12-12-66.mp4    | 12   | 0.79  | 5.89  |
| 动作12-12-67.mp4    | 12   | 0.75  | 6.16  |
| 动作12-13-26.mp4    | 12   | 0.83  | 5.74  |
| 动作13-10-16.mp4    | 13   | 0.54  | 6.24  |
| 动作13-10-30.mp4    | 13   | 0.61  | 5.37  |
| 动作13-10-44.mp4    | 13   | 0.60  | 5.52  |
| 动作13-10-57.mp4    | 13   | 0.68  | 5.27  |
| 动作13-9-70.mp4     | 13   | 0.64  | 5.19  |
| 动作2-4-31.mp4      | 2    | 0.68  | 5.23  |
| 动作2-4-47.mp4      | 2    | 0.74  | 5.06  |
| 动作2-4-5.mp4       | 2    | 0.62  | 5.19  |
| 动作2-4-6.mp4       | 2    | 0.64  | 5.13  |
| 动作3-11-37.mp4     | 3    | 0.80  | 5.77  |
| 动作3-11-38.mp4     | 3    | 0.79  | 6.30  |
| 动作3-12-10.mp4     | 3    | 0.82  | 6.28  |
| 动作3-12-11.mp4     | 3    | 0.81  | 5.92  |
| 动作4-11-40.mp4     | 4    | 0.38  | 8.07  |
| 动作4-11-41.mp4     | 4    | 0.40  | 7.89  |
| 动作4-12-13.mp4     | 4    | 0.39  | 8.73  |
| 动作4-12-14.mp4     | 4    | 0.41  | 7.62  |
| 动作4-12-15.mp4     | 4    | 0.23  | 8.24  |
| 动作5-5-48.mp4      | 5    | 0.54  | 8.25  |
| 动作5-5-60.mp4      | 5    | 0.60  | 8.38  |
| 动作5-5-79.mp4      | 5    | 0.71  | 8.87  |
| 动作5-5-80.mp4      | 5    | 0.39  | 9.78  |
| 动作6-4-13.mp4      | 6    | 0.32  | 5.08  |
| 动作6-4-14.mp4      | 6    | 0.28  | 4.47  |
| 动作6-4-35.mp4      | 6    | 0.34  | 4.58  |
| 动作6-4-55.mp4      | 6    | 0.42  | 5.01  |
| 动作6-4-56.mp4      | 6    | 0.41  | 4.58  |
| 动作6-5-21.mp4      | 6    | 0.38  | 4.60  |
| 动作6-5-35.mp4      | 7    | 0.32  | 4.60  |
| 动作6-5-49.mp4      | 6    | 0.36  | 4.72  |
| 动作6-5-61.mp4      | 6    | 0.43  | 4.64  |
| 动作6-5-7.mp4       | 6    | 0.38  | 4.76  |
| 动作6-5-81.mp4      | 6    | 0.45  | 4.63  |
| 动作7-11-48.mp4     | 7    | 0.99  | 6.79  |
| 动作7-11-49.mp4     | 7    | 0.93  | 6.30  |
| 动作7-11-50.mp4     | 7    | 0.96  | 6.50  |
| 动作7-12-22.mp4     | 7    | 0.92  | 6.32  |
| 动作8-1-23.mp4      | 8    | 0.57  | 8.98  |
| 动作8-1-34.mp4      | 8    | 0.71  | 9.61  |
| 动作8-1-4.mp4       | 8    | 0.75  | 8.74  |
| 动作9-10-12.mp4     | 9    | 0.55  | 8.16  |
| 动作9-10-26.mp4     | 9    | 0.46  | 7.83  |
| 动作9-8-49.mp4      | 9    | 0.61  | 8.50  |

## 二分类任务混淆矩阵

在机器学习中，混淆矩阵是一个非常有用的工具，尤其是在分类问题的评估中。它提供了一个简明扼要的方式来了解模型在不同类别上的表现，尤其是每个类别的预测结果和实际结果之间的关系。

我们分别制作了训练集和验证集上的混淆矩阵图，来直观观察模型的分类效果。从图中可以看出，大多数类别的预测准确度非常高，主对角线上的值接近1.00。这表明模型对这些类别的分类性能非常好。

![训练集混淆矩阵](doc/训练集混淆矩阵.png)
![验证集混淆矩阵](doc/验证集混淆矩阵.png)

## RTMPose 在本项目中的演示

我们简要实现了对 RTMPose 模型输出的关键点 keypoints.npy 数据的可视化，用于直观检查关键点生成效果。可以看到，模型在整个视频时长内都完全准确识别出了人物，并高效、实时地生成了关键点数据。

![RTM效果演示](doc/RTM效果演示.gif)

## 在测试集上的评估效果

### 测试集构建

为了验证模型的泛化效果，需要获取在训练集和验证集上从未出现过的视频数据。通过发动身边的同学和家长，共同参与了构建测试集。

# 模型介绍

[RTMPose介绍.pdf](doc/RTMPose介绍.pdf)

[ST-GCN介绍.pdf](doc/ST-GCN模型介绍.pdf)

[基于距离角度比对的动作打分算法.pdf](基于距离角度比对的动作打分算法.pdf)



# 项目结构

- **/config**: "包含**距离角度判定算法**需要的依赖文件；
- **/data**: 包含模型训练用到的视频关键点 numpy 文件，由 datagen.py 生成，避免每次训练都要重新读取视频；
- **/model**: 存放预训练模型，或模型训练完毕后的默认保存位置；
- **/res**: 存放推理结果，通常为 .csv 文件；
- **/src**: 项目源代码；
- **/tools**: 项目用到的其他工具代码，包括数据结构转换、数据集自动构建、numpy 文件可视化等
- **/vid**: 默认的视频存放路径；
- **/doc**: 存放项目说明文档和相关依赖图片。

一级目录下的其他文件：

- **PoseTracking.py**: 基于预训练模型实现推理和打分任务；
- **README.md**: 本文件；
- **main.py**: 实现 ST-GCN 模型训练和推理功能，已逐步弃用并耦合进 /src；
- **requirements.txt**: 依赖文件列表，不包含 pytorch
- **setup.py**: 自动化部署脚本，待实现


# 部署流程

## 在本地环境克隆或解压此项目

### 使用 git

如果你希望通过 Git 克隆项目，你需要首先确保你的系统已经安装了 Git。可以在终端或命令行中运行以下命令来克隆项目：

```bash
git clone https://github.com/ZHmQAQ/PoseClassifier.git
cd PoseClassifier
```

将 `[项目Git仓库URL]` 替换为你的实际项目 Git 仓库地址。

### 使用压缩包

略。

## 自行安装 PyTorch 相关依赖

根据你的系统和是否需要 GPU 支持，安装命令可能有所不同。你可以在 PyTorch 的官方网站（[PyTorch Get Started](https://pytorch.org/get-started/locally/)）上找到适合你系统的安装指令。

## 安装项目依赖

### 使用 requirements.txt

你可以使用以下命令安装所有必要的 Python 库：

```bash
pip install -r requirements.txt
```

### （可选）安装 ONNX Runtime 相关组件以使用 GPU

如果希望使用 GPU 进行推理，由于 RTMPose 采用 ONNX 模型，因此需要安装以下相关依赖组件：

#### NVidia CUDA Toolkit

1. 访问 [NVIDIA CUDA Toolkit 网站](https://developer.nvidia.com/cuda-downloads)。
2. 选择适合你的操作系统的版本（确保与你的 GPU 驱动兼容）。
3. 下载并安装 CUDA Toolkit。

#### NVidia cuDNN

1. 访问 [NVIDIA cuDNN 页面](https://developer.nvidia.com/cudnn)。
2. 需要注册并登录 NVIDIA 开发者账户以下载 cuDNN。
3. 根据你已安装的 CUDA 版本选择对应的 cuDNN 版本。
4. 下载并按照官方指南安装 cuDNN。

#### ONNX Runtime GPU

安装 ONNX Runtime 以支持 GPU，可以使用 `pip` 直接安装适配 GPU 的版本：

```bash
pip install onnxruntime-gpu
```

