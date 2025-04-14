import torch
import torch.nn as nn
from torchvision.models import resnet101
import torch.nn.functional as F


class DeeplabV3Plus(nn.Module):
   def __init__(self,numclasses=21,output_stride=16):
       super().__init__()
       # 使用ResNet101作为Backone
       backbone = resnet101(pretrained = True)
       # 要根据output_stride去修改空洞层
       if output_stride == 16:
           # 修改layer4
           replace_stride_with_dilation(backbone.layer4,dilation=2)
       elif output_stride == 8:
           # 修改layer3+layer4
           replace_stride_with_dilation(backbone.layer3,dilation=2)
           replace_stride_with_dilation(backbone.layer4,dilation=4)
       self.backbone = nn.Sequential(
           backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
           backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4
       )
       #增加ASPP模块
       self.aspp = ASPP(in_channels=2048,output_stride=output_stride)
       #增加decoder模块
       self.decoder = Decoder(low_level_channels=256,num_classes=numclasses)
   def forward(self,x):
       #提取底层特征,用于decoder模块的输入
       low_level_features = self.backbone[:5](x)
       x = self.backbone[5:](low_level_features)
       # ASPP模块:提取多尺度特征
       x = self.aspp(x)
       # Decoder模块
       x = self.decoder(x,low_level_features)
       # 4倍上采样
       x = F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)
       return x
# 将步幅替换为空洞卷积
def replace_stride_with_dilation(layer,dilation):
    for module in layer.modules():
        if isinstance(module,nn.Conv2d):
            if module.stride == (2,2):
                module.stride = (1,1)
                module.dilation = (dilation,dilation)
                padding = (module.kernel_size[0] // 2) * dilation
                module.padding = (padding,padding)


class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels=256,output_stride=16):
        super().__init__()
        # 空洞率
        dilations = [6,12,18]
        # 根据ASPP的过程:(每个步骤的输入都是原特征图,并不是将前一个输出作为输入)
        # 1.1x1卷积过程:主要提取最原始的局部信息
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 2.3x3空洞(dilation=6)卷积过程:提取小范围的上下文信息
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=dilations[0],dilation=dilations[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 3.3x3空洞(dilation=12)卷积过程:提取中等范围的上下文信息
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=dilations[1],dilation=dilations[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 4.3x3空洞(dilation=18)卷积过程:提取大范围的上下文信息
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=dilations[2],dilation=dilations[2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 5.GAP(全局平均池化):得到全局信息,得到大尺度感受野
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 融合层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels*5,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self,x):
        # 根据ASPP的过程:(每个步骤的输入都是原特征图,并不是将前一个输出作为输入)
        # 1.1x1卷积过程:主要提取最原始的局部信息
        x1 = self.conv1x1(x)
        # 2.3x3空洞(dilation=6)卷积过程:提取小范围的上下文信息
        x2 = self.conv3x3_1(x)
        # 3.3x3空洞(dilation=12)卷积过程:提取中等范围的上下文信息\
        x3 = self.conv3x3_2(x)
        # 4.3x3空洞(dilation=18)卷积过程:提取大范围的上下文信息
        x4 = self.conv3x3_3(x)
        # 5.1 GAP(全局平均池化):得到全局信息,得到大尺度感受野
        x5 = self.global_average_pooling(x)
        # 5.2 线性插值将1x1==>HxW
        x5 = F.interpolate(input=x5,size=x.size()[2:],mode='bilinear',align_corners=True)
        # 6.融合层:融合5层为1层
        x = torch.cat([x1,x2,x3,x4,x5],1)
        x = self.conv_fusion(x)
        return x

class Decoder(nn.Module):
    def __init__(self, low_level_channels,num_classes):
        super().__init__()
        # 处理底层特征信息  压缩通道信息
        self.conv_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels,48,1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        # 处理融合后的信息
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(256+48,256,3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 最终的分类层
        self.classifier = nn.Conv2d(256,num_classes,1)
    def forward(self,x,low_level_features):
        # 处理低层次信息
        low_level_features = self.conv_low_level(low_level_features)
        # 将编码模块得到的高层次信息,上采样到和低层次一样的H,W
        x = F.interpolate(x,size=low_level_features.size()[2:],mode='bilinear',align_corners=True)
        # 合并两类信息
        x =torch.cat([x,low_level_features],dim=1)
        # 处理融合之后的信息
        x =self.conv_fusion(x)
        # 添加分类器
        x = self.classifier(x)
        
        return x
        


