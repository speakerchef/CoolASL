import torch
import torchvision.transforms as transforms
from numpy.f2py.auxfuncs import throw_error
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import os
import yaml
import matplotlib.pyplot as plt

try:
    with open("../configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    throw_error(f"Config not loaded: {e}")

test_path = "data/asl_processed/test"
train_path = "data/asl_processed/train"
asl_test = os.path.join(cfg['working_path'], test_path)
asl_train = os.path.join(cfg['working_path'], train_path)

train_ds = ImageFolder(asl_train, v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Ideal set
test_clean_ds = ImageFolder(asl_test, v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Includes real-world augmentations
test_messy_ds = ImageFolder(asl_test, v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))





train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
test_clean_loader = DataLoader(test_clean_ds, batch_size=cfg['batch_size'], shuffle=True)
test_messy_loader = DataLoader(test_messy_ds, batch_size=cfg['batch_size'], shuffle=True)

# Print sample images
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    class_names = train_ds.classes

    # Unnormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = images[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    

