
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

    # 初始化ResNet-50模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # 加载预训练模型
    model.fc = nn.Linear(model.fc.in_features, 200)  # 修改输出层
    print(model)
    model = model.to(device)

    # 设置优化器，不同层使用不同的学习率
    output_params = list(map(id, model.fc.parameters()))
    features_params = filter(lambda p: id(p) not in output_params, model.parameters())
    optimizer = optim.SGD([
        {'params': features_params, 'lr': learning_rate * 0.1},  # 更低的学习率用于预训练层
        {'params': model.fc.parameters(), 'lr': learning_rate}  # 更高的学习率用于最后的输出层
    ], momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) # 每25步学习率下降0.1

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 模型训练
    trained_model, acc = train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs)
    
    return trained_model, acc


# 参数网格搜索
learning_rates = [0.001, 0.005, 0.01]
num_epochs_options = [50, 100]
best_acc = 0
for lr in learning_rates:
    for epochs in num_epochs_options:
        trained_model, acc = finetune(lr, epochs)

        print(f"Learning Rate: {lr}, Num Epochs: {epochs}, Validation Acc: {acc}")
        with open("param_search_results_resnet50_finetune.txt", "a") as f:
            f.write(f"Learning Rate: {lr}, Num Epochs: {epochs}, Validation Acc: {acc}\n\n")

        if acc > best_acc:
            best_acc = acc
            best_params = {'learning_rate': lr, 'num_epochs': epochs}
            best_model_wts = copy.deepcopy(trained_model.state_dict())

# 保存最佳超参数
print("Best Accuracy:", best_acc)
print("Best Parameters:", best_params)
with open("param_search_results_resnet50_finetune.txt", "a") as f:
    f.write(f"Best Parameters: {best_params}\n")

# 保存最佳模型权重
save_path = "./model_weights"
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save({
    'epoch': best_params["num_epochs"],
    'model_state_dict': best_model_wts,
    'learning_rate': best_params["learning_rate"],
}, os.path.join(save_path, 'ResNet-50_finetune.pth'))
