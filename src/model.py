import torchvision.models
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

model = resnet18(weights=ResNet18_Weights.DEFAULT)

features_in = model.fc.in_features
model.fc = nn.Linear(features_in, 36) # A-Z, 0-9


