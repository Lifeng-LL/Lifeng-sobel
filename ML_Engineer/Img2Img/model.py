import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelNet(nn.Module):
    def __init__(self):
        super(SobelNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect", bias=True)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect", bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.relu(self.conv1(img))
        img = self.relu(self.conv2(img))
        return img
