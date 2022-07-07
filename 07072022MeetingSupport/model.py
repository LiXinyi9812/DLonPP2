import torch.nn as nn
from torchvision import models
import torch

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

def create_vgg19(model_vgg_path):
    model = models.vgg19(pretrained = False)
    model.load_state_dict(torch.load(model_vgg_path))
    return model

def create_DRAMA(device):
    drama_model = [DRAMA().to(device), DRAMA().to(device), DRAMA().to(device), DRAMA().to(device), DRAMA().to(device),
                  DRAMA().to(device)]
    return drama_model