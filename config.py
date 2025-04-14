import torch

class Config:
    # 超参数
    NUM_CLASSES = 21
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    IMG_SIZE = (256, 256)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径：请修改为你本地 VOC2012 的路径（不要写到 JPEGImages，要写到 VOCdevkit 那一层）
    VOC_ROOT = r"E:\Works\datasets"  # 🧠改成你本地的 VOC 根目录！！！
