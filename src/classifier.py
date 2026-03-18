import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

from src.alexnet import AlexNet

class CardClassifier(nn.Module):
    def __init__(self, arch_name, num_classes, pretrained=False):
        super(CardClassifier, self).__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.arch_name = arch_name
        mode = "pretrained weights" if pretrained else "random initialization"

        if arch_name == 'alexnet':
            self.model = AlexNet(num_classes=num_classes)
            self.apply(self.init_weights)
            print("Loaded custom AlexNet (training from scratch)")

        elif arch_name == 'resnet':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            print(f"=> Loaded ResNet18 ({mode})")

        elif arch_name == 'mobilenet':
            weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
            print(f"=> Loaded MobileNetV2 ({mode})")
        
        else:
            raise ValueError(f"Architecture {arch_name} is not supported.")

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)