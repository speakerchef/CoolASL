import torch
from configs.cfg_importer import *
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt

cfg = import_cfg()

test_path = "data/asl_processed/test"
train_path = "data/asl_processed/train"
asl_test = os.path.join(cfg['working_path'], test_path)
asl_train = os.path.join(cfg['working_path'], train_path)

transforms = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.RandomRotation(15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = ImageFolder(asl_train, transforms)

# Ideal set
test_clean_ds = ImageFolder(asl_test, v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Includes real-world augmentations
test_messy_ds = ImageFolder(asl_test, transforms)





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


