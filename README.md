# CUB-classification
本仓库通过微调在ImageNet上预训练的ResNet-18和ResNet-50模型，实现对于CUB-200-2011数据集的鸟类识别分类。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据读取与预处理

在 [`utils/dataset_loader.py`](utils/dataset_loader.py) 中定义的`load_dataset`函数用于加载数据集，并自动根据CUB-200-2011数据集中的`train_test_split.txt`文件划分为训练集和测试集，同时进行数据增强处理，在调用时可指定数据集的batch size。

`CUB_200_2011.tgz`数据文件需要直接存放在本项目的根目录。在初次加载数据时，将会新建`processed`目录，并生成`processed/train.pkl`和`processed/test.pkl`。

## 微调预训练模型

运行 [`finetune_resnet18.py`](finetune_resnet18.py) 即可对ResNet-18模型进行微调训练，同理运行[`finetune_resnet50.py`](finetune_resnet50.py) 即可对ResNet-50模型进行微调训练。

在微调训练过程中，将自动执行超参数网格搜索，可指定学习率和训练的epoch数量的备选值，例如
```python
learning_rates = [0.001, 0.005, 0.01]
num_epochs_options = [50, 100]
```

最终将保存最佳的超参数设置，以及最佳的模型权重。对于ResNet-18模型，超参数搜索结果将保存至根目录下的`param_search_results_resnet18_finetune.txt`，模型权重保存至`weights/ResNet-18_finetune.pth`；对于ResNet-50模型，超参数搜索结果将保存至根目录下的`param_search_results_resnet50_finetune.txt`，模型权重保存至`weights/ResNet-50_finetune.pth`。

训练过程的Tensorboard记录将保存至自动新建的`runs`目录下。

具体模型训练结果请参见实验报告。

## 参数随机初始化的模型训练

运行 [`trainNew_resnet18.py`](trainNew_resnet18.py) 可从随机初始化的网络参数开始训练ResNet-18模型；运行 [`trainNew_resnet50.py`](trainNew_resnet50.py) 可从随机初始化的网络参数开始训练ResNet-50模型。

具体模型训练结果请参见实验报告。