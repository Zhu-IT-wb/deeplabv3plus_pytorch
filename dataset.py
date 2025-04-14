from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms.functional import InterpolationMode
from config import Config

# 图像 & 标签预处理
transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

# 加载数据
def get_dataloaders():
    train_set = VOCSegmentation(
        root=Config.VOC_ROOT,
        year='2012',
        image_set='train',
        download=False,
        transform=transform,
        target_transform=target_transform
    )
    val_set = VOCSegmentation(
        root=Config.VOC_ROOT,
        year='2012',
        image_set='val',
        download=False,
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader
