
from utils.dataset_loader import load_dataset
from utils.model_trainer import train_model
import os
import copy
import torch
from torch import optim, nn
from torchvision import models


train_dataset, test_dataset, train_dataloader, test_dataloader = load_dataset(batch_size=8)
dataloaders = {"train": train_dataloader, "val": test_dataloader}
dataset_sizes = {"train": len(train_dataset), "val": len(test_dataset)}

def finetune(learning_rate, num_epochs):
    # 检查GPU可用性并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.backends.cudnn.enabled = False

    # 初始化ResNet-18模型
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 200)  # 修改输出层
    print(model)
    model = model.to(device)

    # 设置优化器
    optimizer = optim.SGD([{'params': model.parameters(), 'lr': learning_rate}], momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) # 每25步学习率下降0.1

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 模型训练
    trained_model, acc = train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs)
    
    return trained_model, acc


# 设置使用与微调相同的超参数，训练模型
lr = 0.001
epochs = 100
trained_model, acc = finetune(lr, epochs)
best_model_wts = copy.deepcopy(trained_model.state_dict())

# 保存模型权重
save_path = "./model_weights"
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save({
    'epoch': epochs,
    'model_state_dict': best_model_wts,
    'learning_rate': lr,
}, os.path.join(save_path, 'ResNet-18_trainNew.pth'))
