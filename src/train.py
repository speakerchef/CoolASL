import datetime

import torch
import torch.nn as nn

from configs.cfg_importer import import_cfg
from src.dataloader import train_loader, test_messy_loader, test_clean_loader

from src.model import model

import matplotlib.pyplot as plt

import json

cfg = import_cfg()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)

# Parameters
num_epochs = cfg['num_epochs']
lr = cfg['lr']
optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr)
criterion = nn.CrossEntropyLoss()

def train_resnet18():
    """Train for 1 epoch"""
    running_loss = 0.0
    num_samples = 0
    for i, data in enumerate(train_loader):
        img, labels = data
        img, labels = img.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(img)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * img.size(0)
        num_samples += img.size(0)

    return running_loss/num_samples # Loss per epoch


# Training loop
train_losses = []
val_losses = []
date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
print(f'date: {date}')
for epoch in range(num_epochs):
    print(f"Training Epoch: {epoch + 1}")

    model.train()
    train_loss = train_resnet18()
    train_losses.append(train_loss)

    running_vloss = 0.0
    num_vsamples = 0
    model.eval()

    # Validation loop
    with torch.no_grad():
        for j, vdata in enumerate(test_clean_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            voutputs = model(vinputs)

            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item() * vinputs.size(0)
            num_vsamples += vinputs.size(0)

    val_loss = running_vloss/num_vsamples
    val_losses.append(val_loss)

    # Save metrics for future reference
    with open(f"../logs/model-metrics/model_metrics_{date}.json", 'w') as f:
        json.dump({"epoch": epoch, "training_loss": train_losses, "validation_loss": val_losses}, f)

    # Save for future access
    torch.save(model.state_dict(), f'../models/resnet18_asl_{date}')

    print(f"Training Loss: {train_loss}, Validation Loss {val_loss}")


# Plot losses
