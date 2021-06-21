import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import os
from torch.nn import init
import torch.backends.cudnn as cudnn

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)

class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class ModelHead(nn.Module):
    def __init__(self, in_channels=6):
        super(ModelHead, self).__init__()

        self.inc = inconv(in_channels, 64)
        self.down1 = Downconv(64, 128)
        self.down2 = Downconv(128, 128)
        self.down3 = Downconv(128, 128)

    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)

        return catfeat, dx4

class ModelTail(nn.Module):
    def __init__(self, in_channels=128, momentum=0.1):
        super(ModelTail, self).__init__()

        self.momentum = momentum

        self.features = nn.Sequential(
            conv_block(0, in_channels=in_channels, out_channels=128, momentum=self.momentum, pooling=True),
            conv_block(1, in_channels=128, out_channels=256, momentum=self.momentum, pooling=True),
            conv_block(2, in_channels=256, out_channels=512, momentum=self.momentum, pooling=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.add_module('fc', nn.Linear(512, 1))

    def forward(self, x, params=None):
        if params == None:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            out = F.conv2d(
                x,
                params['features.0.conv0.weight'],
                params['features.0.conv0.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.0.bn0.running_mean'],
                params['features.0.bn0.running_var'],
                params['features.0.bn0.weight'],
                params['features.0.bn0.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)

            out = F.conv2d(
                out,
                params['features.1.conv1.weight'],
                params['features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.1.bn1.running_mean'],
                params['features.1.bn1.running_var'],
                params['features.1.bn1.weight'],
                params['features.1.bn1.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)

            out = F.conv2d(
                out,
                params['features.2.conv2.weight'],
                params['features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.2.bn2.running_mean'],
                params['features.2.bn2.running_var'],
                params['features.2.bn2.weight'],
                params['features.2.bn2.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, 1)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['fc.weight'],
                           params['fc.bias'])
        return out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict


def conv_block(index,
               in_channels,
               out_channels,
               K_SIZE=3,
               stride=1,
               padding=1,
               momentum=0.1,
               pooling=True):
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv' + str(index), nn.Conv2d(
                    in_channels, out_channels, K_SIZE, stride=stride, padding=padding)),
                ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum, affine=True)),
                ('relu' + str(index), nn.ReLU(inplace=True)),
                ('pool' + str(index), nn.MaxPool2d(2))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv' + str(index), nn.Conv2d(in_channels, out_channels, K_SIZE, padding=padding)),
                ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum,  affine=True)),
                ('relu' + str(index), nn.ReLU(inplace=True))
            ]))
    return conv


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 1.02)
        init.constant_(m.bias.data, 0.0)



def init_model(net, restore, parallel_reload=True):
    """Init models with cuda and weights."""
    net.apply(weights_init_xavier)

    if restore is not None and os.path.exists(restore):
        print("*************Restore model from: {}".format(os.path.abspath(restore)))
        state_dict = torch.load(restore)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if parallel_reload:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.restored = True

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net
