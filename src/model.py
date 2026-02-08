import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

features_in = model.fc.in_features
model.fc = nn.Linear(features_in, 36) # A-Z, 0-9

# Freezing early layers to avoid overfitting.
for name, param in model.named_parameters():
    if name.startswith(('layer1', 'layer2', 'layer3')):
        param.requires_grad = False







