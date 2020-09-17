import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal


class NoisyActivation(nn.Module):
    """
    new method to pre-process the training data
    """

    def __init__(self, shape, device, upper_bound, threshold):
        super(NoisyActivation, self).__init__()
        self.shape = shape
        self.device = device
        self.upper_bound = upper_bound
        self.threshold = threshold
        self.mus = nn.Parameter(nn.init.normal_(torch.empty(self.shape, device=self.device), 0, 0.2))
        self.rhos = nn.Parameter(nn.init.normal_(torch.empty(self.shape, device=self.device), 0, 1))
        self.sigma = (1 + torch.tanh(self.rhos)) / 2 * self.upper_bound

        self.normal = Normal(0, 1)

    def forward(self, x):
        """
        :param x: input
        :return:
        """
        self.sigma = (1 + torch.tanh(self.rhos)) / 2 * self.upper_bound
        noise = self.sigma * (self.normal.sample(self.shape).to(self.device)) + self.mus
        return x + noise

    def inner_loss(self):
        """
        return -log((1/n)*\\sum{sigma**2})
        """
        return -torch.log(1 / np.prod(self.shape) * torch.sum(torch.pow(self.sigma, 2)))


def get_model(model_index, num_classes):
    """
    get PyTorch models

    :param model_index: from 0 to 37
    :param num_classes:
    0 alexnet
    1 DenseNet121
    2 DenseNet161
    3 DenseNet169
    4 DenseNet201
    5 ge_resnext29_8x64d
    6 ge_resnext29_16x64d
    7 google_net
    8 lenet
    9 preresnet20
    10 preresnet32
    11 preresnet44
    12 preresnet56
    13 preresnet110
    14 preresnet1202
    15 RegNetX_200MF
    16 RegNetX_400MF
    17 RegNetY_400MF
    18 resnet20
    19 resnet32
    20 resnet44
    21 resnet56
    22 resnet110
    23 resnet1202
    24 ResNeXt29_2x64d
    25 ResNeXt29_4x64d
    26 ResNeXt29_8x64d
    27 ResNeXt29_32x4d
    28 se_resnext29_8x64d
    29 se_resnext29_16x64d
    30 sk_resnext29_16x32d
    31 sk_resnext29_16x64d
    32 vgg11
    33 vgg13
    34 vgg16
    35 vgg19
    """
    model = None
    model_name = None
    try:
        if model_index == 0:
            from models.alexnet import alexnet
            model = alexnet(num_classes)
            model_name = 'alexnet'
        elif model_index == 1:
            from models.densenet import DenseNet121
            model = DenseNet121(num_classes)
            model_name = 'DenseNet121'
        elif model_index == 2:
            from models.densenet import DenseNet161
            model = DenseNet161(num_classes)
            model_name = 'DenseNet161'
        elif model_index == 3:
            from models.densenet import DenseNet169
            model = DenseNet169(num_classes)
            model_name = 'DenseNet169'
        elif model_index == 4:
            from models.densenet import DenseNet201
            model = DenseNet201(num_classes)
            model_name = 'DenseNet201'
        elif model_index == 5:
            from models.genet import ge_resnext29_8x64d
            model = ge_resnext29_8x64d(num_classes)
            model_name = 'ge_resnext29_8x64d'
        elif model_index == 6:
            from models.genet import ge_resnext29_16x64d
            model = ge_resnext29_16x64d(num_classes)
            model_name = 'ge_resnext29_16x64d'
        elif model_index == 7:
            from models.googlenet import google_net
            model = google_net(num_classes)
            model_name = 'google_net'
        elif model_index == 8:
            from models.lenet import lenet
            model = lenet(num_classes)
            model_name = 'lenet'
        elif model_index == 9:
            from models.preresnet import preresnet20
            model = preresnet20(num_classes)
            model_name = 'preresnet20'
        elif model_index == 10:
            from models.preresnet import preresnet32
            model = preresnet32(num_classes)
            model_name = 'preresnet32'
        elif model_index == 11:
            from models.preresnet import preresnet44
            model = preresnet44(num_classes)
            model_name = 'preresnet44'
        elif model_index == 12:
            from models.preresnet import preresnet56
            model = preresnet56(num_classes)
            model_name = 'preresnet56'
        elif model_index == 13:
            from models.preresnet import preresnet110
            model = preresnet110(num_classes)
            model_name = 'preresnet110'
        elif model_index == 14:
            from models.preresnet import preresnet1202
            model = preresnet1202(num_classes)
            model_name = 'preresnet1202'
        elif model_index == 15:
            from models.regnet import RegNetX_200MF
            model = RegNetX_200MF(num_classes)
            model_name = 'RegNetX_200MF'
        elif model_index == 16:
            from models.regnet import RegNetX_400MF
            model = RegNetX_400MF(num_classes)
            model_name = 'RegNetX_400MF'
        elif model_index == 17:
            from models.regnet import RegNetY_400MF
            model = RegNetY_400MF(num_classes)
            model_name = 'RegNetY_400MF'
        elif model_index == 18:
            from models.resnet import resnet20
            model = resnet20(num_classes)
            model_name = 'resnet20'
        elif model_index == 19:
            from models.resnet import resnet32
            model = resnet32(num_classes)
            model_name = 'resnet32'
        elif model_index == 20:
            from models.resnet import resnet44
            model = resnet44(num_classes)
            model_name = 'resnet44'
        elif model_index == 21:
            from models.resnet import resnet56
            model = resnet56(num_classes)
            model_name = 'resnet56'
        elif model_index == 22:
            from models.resnet import resnet110
            model = resnet110(num_classes)
            model_name = 'resnet110'
        elif model_index == 23:
            from models.resnet import resnet1202
            model = resnet1202(num_classes)
            model_name = 'resnet1202'
        elif model_index == 24:
            from models.resnext import ResNeXt29_2x64d
            model = ResNeXt29_2x64d(num_classes)
            model_name = 'ResNeXt29_2x64d'
        elif model_index == 25:
            from models.resnext import ResNeXt29_4x64d
            model = ResNeXt29_4x64d(num_classes)
            model_name = 'ResNeXt29_4x64d'
        elif model_index == 26:
            from models.resnext import ResNeXt29_8x64d
            model = ResNeXt29_8x64d(num_classes)
            model_name = 'ResNeXt29_8x64d'
        elif model_index == 27:
            from models.resnext import ResNeXt29_32x4d
            model = ResNeXt29_32x4d(num_classes)
            model_name = 'ResNeXt29_32x4d'
        elif model_index == 28:
            from models.senet import se_resnext29_8x64d
            model = se_resnext29_8x64d(num_classes)
            model_name = 'se_resnext29_8x64d'
        elif model_index == 29:
            from models.senet import se_resnext29_16x64d
            model = se_resnext29_16x64d(num_classes)
            model_name = 'se_resnext29_16x64d'
        elif model_index == 30:
            from models.sknet import sk_resnext29_16x32d
            model = sk_resnext29_16x32d(num_classes)
            model_name = 'sk_resnext29_16x32d'
        elif model_index == 31:
            from models.sknet import sk_resnext29_16x64d
            model = sk_resnext29_16x64d(num_classes)
            model_name = 'sk_resnext29_16x64d'
        elif model_index == 32:
            from models.vgg import vgg11
            model = vgg11(num_classes)
            model_name = 'vgg11'
        elif model_index == 33:
            from models.vgg import vgg13
            model = vgg13(num_classes)
            model_name = 'vgg13'
        elif model_index == 34:
            from models.vgg import vgg16
            model = vgg16(num_classes)
            model_name = 'vgg16'
        elif model_index == 35:
            from models.vgg import vgg19
            model = vgg19(num_classes)
            model_name = 'vgg19'
    except IndexError:
        print('IndexError: model_index should ba an integar between 0 to 35.')
    return model, model_name
