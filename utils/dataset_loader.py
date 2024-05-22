from cub import cub200
from torchvision import transforms
from torch.utils.data import DataLoader

def load_dataset(batch_size = 16):
    # 读入数据集
    path = '/remote-home/codes_cub'

    IMAGE_SIZE = 224
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD = [0.229, 0.224, 0.225]
    TEST_MEAN = [0.485, 0.456, 0.406]
    TEST_STD = [0.229, 0.224, 0.225]

    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(TEST_MEAN, TEST_STD)
    ])

    train_dataset = cub200(
        path,
        train=True,
        transform=train_transform
    )
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = cub200(
        path,
        train=False,
        transform=test_transform
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    

    print(f"train data size: {len(train_dataset)}")
    print(f"test data size: {len(test_dataset)}")
    # print(len(train_dataloader))
    # print(len(test_dataloader))
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader
