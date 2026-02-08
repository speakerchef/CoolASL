import torch.nn as nn
from dataloader import *
from model import *
from ignite.handlers import Engine, Events
from ignite.metrics import Accuracy, Loss

num_epochs = 50 # For now
lr = 0.0001
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()





def train():
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        input, label = data

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

