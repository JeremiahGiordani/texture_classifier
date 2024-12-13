import torch
import torch.nn as nn
import torchvision.models as models

pretrained_weights = "/home/jg0037/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

def get_pretrained_model(num_classes=4):
    # Use ResNet18 pretrained on ImageNet
    weights = torch.load(pretrained_weights)
    model = models.resnet18()
    model.load_state_dict(weights)
    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model
