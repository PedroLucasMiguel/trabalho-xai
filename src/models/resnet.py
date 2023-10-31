import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class Resnet50GradCam(nn.Module):
    def __init__(self, backbone:nn.Module, n_classes:int) -> None:
        super().__init__()

        self.backbone = backbone
        self.backbone.fc = nn.Linear(2048, n_classes)

        self.gradient = None

    def gradient_hook(self, gradient):
        self.gradient = gradient

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x:Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        if x.requires_grad:
            h = x.register_hook(self.gradient_hook)

        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x
    