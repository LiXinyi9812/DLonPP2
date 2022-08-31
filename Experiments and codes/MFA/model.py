from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import misc as misc_nn_ops
from memory_profiler import profile as profile_cpu
from line_profiler import LineProfiler

torch.set_default_tensor_type(torch.DoubleTensor)
@torch.no_grad()
def get_backbone_output_shape(backbone, input_shape=(3, 224, 224)):
    input_tensor = torch.Tensor(*input_shape)
    output = backbone(input_tensor.unsqueeze(0)).squeeze(0)
    return tuple(output.shape)


def get_resnet50_backbone(trainable_layers=3):
    """Return a trained ResNet50 without the last avgpool and fc layers and the shape of the output feature map with an
    input of shape (300, 400) corresponding to an image from the PP2 dataset. All layers are frozen except the last
    `trainable_layers` layers (see https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py).
    :param trainable_layers: int between 0 (freeze everything) and 5 (train everything)
    :return:
    """
    # remove the last layers of resnet to make a feature extractor
    # avgpool is removed since we add convolutions after the feature extraction is SSCNN
    resnet = torchvision.models.resnet.__dict__['resnet50'](
        pretrained=True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    modules = OrderedDict(resnet.named_children())
    del modules['avgpool']
    del modules['fc']
    feature_extractor = nn.Sequential(modules)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    for name, parameter in feature_extractor.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return feature_extractor


def get_alexnet_backbone(trainable_layers=3):
    """Return a trained Alexnet network without the avgpool and classifier layers and the shape of the output feature map
    with an input of shape (300, 400) corresponding to an image from the PP2 dataset.
    :param trainable_layers: int between 0 (freeze everything) and 5 (train everything)
    :return:
    """
    alexnet = torchvision.models.alexnet(pretrained=True)
    modules = OrderedDict(alexnet.named_children())
    del modules['avgpool']
    del modules['classifier']
    feature_extractor = nn.Sequential(modules)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['features.10', 'features.8', 'features.6', 'features.3', 'features.0'][:trainable_layers]
    for name, parameter in feature_extractor.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return feature_extractor


def get_vgg_backbone(version, trainable_layers=3):
    """Return a trained VGG16 network without the avgpool and classifier layers and the shape of the output feature map
    with an input of shape (300, 400) corresponding to an image from the PP2 dataset.
    :param int version: 16 (VGG16) or 19 (VGG19)
    :param trainable_layers: int between 0 (freeze everything) and 5 (train everything)
    :return:
    """
    assert version in (16, 19)
    vgg_models = {
        16: torchvision.models.vgg16,
        19: torchvision.models.vgg19
    }
    vgg = vgg_models[version](pretrained=True)
    modules = OrderedDict(vgg.named_children())
    del modules['avgpool']
    del modules['classifier']
    feature_extractor = nn.Sequential(modules)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = _get_vgg_trainable_layers_names(version, trainable_layers)
    for name, parameter in feature_extractor.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return feature_extractor


def _get_vgg_trainable_layers_names(version, trainable_layers):
    vgg_layers = {
        16: ((24, 26, 28), (17, 19, 21), (10, 12, 14), (5, 7), (0, 2)),
        19: ((28, 30, 32, 34), (19, 21, 23, 25), (10, 12, 14, 16), (5, 7), (0, 2))
    }
    trainable = vgg_layers[version][:trainable_layers]
    trainable = tuple(reduce(lambda x, y: x+y, trainable, tuple()))  # flatten the tuple
    return ['features.{}'.format(k) for k in trainable]


def get_backbone(name, input_shape=(3, 224, 224), trainable_layers=0):
    """Return the pretrained backbone (nn.Module) and the shape of the output feature map with an
    input of shape (3, 224, 224), corresponding to an image from the PP2 dataset. The default input shape is the
    standard one use for backbones trained on ImageNet.
    """
    if name == 'resnet50':
        backbone = get_resnet50_backbone(trainable_layers)
    elif name == 'alexnet':
        backbone = get_alexnet_backbone(trainable_layers)
    elif name == 'vgg16':
        backbone = get_vgg_backbone(16, trainable_layers)
    elif name == 'vgg19':
        backbone = get_vgg_backbone(19, trainable_layers)
    else:
        raise ValueError('{} is currently not supported as backbone'.format(name))

    return backbone, get_backbone_output_shape(backbone, input_shape)

class info_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(66, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 6)
        )
    def forward(self, x):
        x = self.extractor(x)
        return x


class SSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(6, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, left, right):
        x = self.fuse(left, right)
        return x  # do not normalize in training as we use BCEWithLogitsLoss

    def fuse(self, f1, f2):
        x = f1-f2
        pred = self.fusion(x)
        return pred


class RSSNN(nn.Module):
    def __init__(self, sscnn):
        super().__init__()
        self.sscnn = sscnn
        self.ranking = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1)
        )

    def forward(self, left, right):
        sscnn_pred = self.sscnn.fuse(left, right)
        rssccn_pred_left = self.ranking(left)
        rssccn_pred_right = self.ranking(right)
        return sscnn_pred, rssccn_pred_left, rssccn_pred_right

def create_RSSNN(atts, device):
    rssnn_model = dict(zip(atts, [RSSNN(SSNN()).double().to(device)] * len(atts)))
    return rssnn_model


class DRAMA(nn.Module):
    def __init__(self):
        super(DRAMA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 10,kernel_size=3)
        self.relu1 = nn.Softplus()
        self.conv2 = nn.Conv2d(in_channels=10,out_channels= 10,kernel_size=3)
        self.relu2 = nn.Softplus()
        self.conv3 = nn.Conv2d(in_channels=10,out_channels= 10,kernel_size=3)
        self.relu3 = nn.Softplus()
        self.dense = nn.Linear(in_features=10*43*43,out_features=1*2)
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.relu1(x_1)
        x_3 = self.conv2(x_2)
        x_4 = self.relu2(x_3)
        x_5 = self.conv3(x_4)
        x_6 = self.relu3(x_5)
        x_7 = x_6.view(x_6.size(0),10*43*43)
        res = self.dense(x_7)
        return  res,self.dense.weight.data


def create_DRAMA(atts, device):
    drama_model = dict(zip(atts, [DRAMA().to(device)] * len(atts)))
    return drama_model



class fusion_net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, left, right,batch_size):
        f1, f2 = self.extract_features(left, right)
        x = self.fuse(f1, f2,batch_size)
        return x  # do not normalize in training as we use BCEWithLogitsLoss

    def extract_features(self, left, right):
        f1 = self.backbone(left)
        f2 = self.backbone(right)
        return f1, f2

    def fuse(self, f1, f2,batch_size):
        f1 = (f1.reshape((batch_size,512,-1))).squeeze()
        f2 = (f2.reshape((batch_size, 512, -1))).squeeze()
        f1 = f1.permute(0,2,1)
        x = (torch.bmm(f1, f2)).unsqueeze(dim=1)  # cross product
        return x


