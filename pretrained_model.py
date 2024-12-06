import torch
import torch.nn as nn
import torchvision.models as models

def get_pretrained_model(num_classes=4):
    # Example: Use ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
