import torchvision.models as models
import torch.nn as nn


# Simple Landmark Classifier (New)
# Improve on Resnet IC method by bypassing domain gap between dataset and live feed;
# Image background, color, hue, etc... don't matter.
class LandmarkClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(42, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.network(x)



# Resnet18 (deprecated)
# Performs poorly on real world material due to large variances in image backgrounds;
# **Training images have consistent, clean backgrounds.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

features_in = model.fc.in_features
model.fc = nn.Linear(features_in, 26) # A-Z

# Freezing early layers to avoid overfitting.
for name, param in model.named_parameters():
    if name.startswith(('layer1', 'layer2', 'layer3')):
        param.requires_grad = False








