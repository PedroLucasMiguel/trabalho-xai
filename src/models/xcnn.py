import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class XCNN(nn.Module):

    def __init__(self, backbone:nn.Module, n_classes:int=2) -> None:
        super().__init__()

        self.gradient = None

        # Enconder-decoder model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 3, 3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.backbone = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.classifier[6] = nn.Linear(4096, n_classes)
        #self.backbone.classifier[6] = nn.Linear(4096, n_classes)

    def gradient_hook(self, gradient):
        self.gradient = gradient

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x:Tensor):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.backbone(out)

        return out

    def forward(self, x:Tensor) -> Tensor:
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.backbone(out)

        out = self.avgpool(out)

        if out.requires_grad:
            h = out.register_hook(self.gradient_hook)

        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out