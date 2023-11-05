"""
@Project ：coding 
@File    ：ResNet_Pre.py
@IDE     ：PyCharm 
@Author  ：ZChang
@Date    ：2023/4/19 11:11 
"""
import torch.nn as nn
import torch
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Encoder import ResNet34Encoder
from wama_modules.Head import ClassificationHead
from wama_modules.BaseModule import GlobalAvgPool
import torch
import torch.nn as nn
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Head import ClassificationHead
from wama_modules.BaseModule import GlobalMaxPool

"""You can download the code from this link"""
"""https://github.com/WAMAWAMA"""
class Resnet2d(nn.Module):
    def __init__(self, in_channel, label_category_dict, dim=2):
        super().__init__()
        # encoder
        f_channel_list = [64, 128, 256, 512]
        self.encoder = ResNetEncoder(
            in_channel,
            blocks=[3, 4, 6, 3],
            type='33',
            downsample_ration=[0.5, 0.5, 0.5, 0.5],
            dim=dim)
        # cls head
        self.cls_head = ClassificationHead(label_category_dict, f_channel_list[-1])
        self.pooling = GlobalAvgPool()

    def forward(self, x):
        f = self.encoder(x)
        feature = self.pooling(f[-1])
        logits = self.cls_head(self.pooling(f[-1]))
        return feature, logits

class Resnet2d50(nn.Module):
    def __init__(self, in_channel, label_category_dict, dim=2):
        super().__init__()
        # encoder
        # f_channel_list = [64, 128, 256, 512]
        self.encoder = ResNetEncoder(
            in_channel,
            stage_output_channels=[256, 512, 1024, 2048],
            stage_middle_channels=[64, 128, 256, 512],
            blocks=[3, 4, 6, 3],
            type='131',
            downsample_ration=[0.5, 0.5, 0.5, 0.5],
            dim=dim)
        # cls head
        self.cls_head = ClassificationHead(label_category_dict, 2048)
        self.pooling = GlobalAvgPool()

    def forward(self, x):
        f = self.encoder(x)
        feature = self.pooling(f[-1])
        logits = self.cls_head(self.pooling(f[-1]))
        return logits


if __name__ == '__main__':
    x = torch.ones([24, 3, 128, 128])
    label_category_dict = dict(is_malignant=2)
    model = Resnet2d50(in_channel=3, label_category_dict=label_category_dict, dim=2)
    logits = model(x)
    print('single-label predicted logits')
    _ = [print('logits of ', key, ':', logits[key].shape) for key in logits.keys()]
    # _ = [print()]
    for key in model.state_dict().keys():
        print(key)
        if 'bn' in key:
            print('bn in key')
