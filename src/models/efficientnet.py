import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from torch import Tensor

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

TRAIN_URL = "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"

class EfficientNetB0GradCam(nn.Module):
    def __init__(self, backbone:nn.Module, n_classes:int) -> None:
        super().__init__()
        self.backbone = backbone

        # Workaround relativo a problemas de hash ao carregar o treinamento
        backbone.load_state_dict(torch.hub.load_state_dict_from_url(TRAIN_URL, progress=False))

        self.backbone.classifier[1] = nn.Linear(1280, n_classes)

        self.gradient = None

    def gradient_hook(self, gradient):
        self.gradient = gradient

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x:Tensor):
        x = self.backbone.features(x)

        return x
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        
        if x.requires_grad:
            h = x.register_hook(self.gradient_hook)

        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        return x

if __name__ == "__main__":
    print(EfficientNetB0GradCam(efficientnet_b0(), 2))