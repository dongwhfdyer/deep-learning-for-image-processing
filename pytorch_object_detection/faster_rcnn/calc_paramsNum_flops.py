import os
import datetime

import torch
import torchvision
from ptflops import get_model_complexity_info
from torchstat import stat
import torchsummary
from thop import profile
from thop import clever_format
import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone

if __name__ == '__main__':
    backbone = resnet50_fpn_backbone(
        norm_layer=torch.nn.BatchNorm2d,
        trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    # net = FasterRCNN(backbone=backbone, num_classes=2)
    # get a resnet model from torchvision
    net = torchvision.models.resnet50()

    ################################################## stat
    # stat(net, [3, 608, 608])
    ##################################################

    # ################################################## calculate the params
    torchsummary.summary(net.cuda(), (3, 608, 608))
    # ##################################################

    ################################################## thop
    myinput = torch.zeros((1, 3, 608, 608)).to('cuda')
    flops, params = profile(net.to('cuda'), inputs=(myinput,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    ##################################################

    ################################################## ptflops
    with torch.cuda.device(0):
        net = retinanet(2, phi=2)
        macs, params = get_model_complexity_info(net,
                                                 (3, 608, 608),
                                                 # (3, 224, 224),
                                                 as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    ##################################################
