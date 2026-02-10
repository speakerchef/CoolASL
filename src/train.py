import datetime
import logging as log
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from configs.cfg_importer import import_cfg
from src.dataloader import init_resnet18
from src.model import resnet18, LandmarkClassifier
import json

cfg = import_cfg()

lm_train_path = cfg['ASLHG_LM_TRAIN']
lm_test_path = cfg['ASLHG_LM_TEST']

MODEL_MODE = cfg['MODEL_MODE']

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")
log.info(f"Model: {MODEL_MODE}")
lmc_model = LandmarkClassifier()
resnet18_model = resnet18.to(device)
lmc_model.to(device)

# Parameters
num_epochs = cfg['num_epochs']
lr = cfg['lr']

optimizer_resnet18 = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, resnet18_model.parameters()),
    lr=lr)

optimizer_lmc = torch.optim.Adam(params=lmc_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def train_loop(model, t_loader, optimizer):
    """
    Train model for one epoch
    :param model: Chosen model
    :param t_loader: train dataloader
    :param optimizer: Chosen optimizer
    :return: Loss per epoch
    """

    running_loss = 0.0
    num_samples = 0
    model.train()
    for i, data in enumerate(t_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    return running_loss/num_samples # Loss per epoch

@torch.no_grad()
def eval_loop(model, val_loader):
    """
    Evaluates the performance of a model during inference on test/unseen data for one epoch
    :param model: Your chosen model
    :param val_loader: Dataloader for test dataset
    :return: Loss per epoch
    """
    running_vloss = 0.0
    num_vsamples = 0
    model.eval()

    # Validation loop
    for j, vdata in enumerate(val_loader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)

        voutputs = model(vinputs)

        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss.item() * vinputs.size(0)
        num_vsamples += vinputs.size(0)

    return running_vloss / num_vsamples


def train_lmc():
    """
    Trains the LandmarkClassifier model
    """
    data_tr = torch.load(lm_train_path)
    data_tst = torch.load(lm_test_path)
    classes = sorted(data_tr.keys())


    all_landmarks_tr = []
    all_landmarks_tst = []
    labels_tr = []
    labels_tst = []

    # Creates label and input lists
    for cls_idx, cls_name in enumerate(classes):
        for coords in data_tr[cls_name]:
            all_landmarks_tr.append(coords)
            labels_tr.append(cls_idx)
        for coords in data_tst[cls_name]:
            all_landmarks_tst.append(coords)
            labels_tst.append(cls_idx)


    # Prep raw data for training
    lm_train_ds = TensorDataset(torch.tensor(all_landmarks_tr, dtype=torch.float32),
                                torch.tensor(labels_tr, dtype=torch.long))
    lm_train_loader = DataLoader(lm_train_ds, batch_size=cfg['batch_size'], shuffle=True)
    lm_test_ds = TensorDataset(torch.tensor(all_landmarks_tst, dtype=torch.float32),
                               torch.tensor(labels_tst, dtype=torch.long))
    lm_test_loader = DataLoader(lm_test_ds, batch_size=cfg['batch_size'], shuffle=True)

    train_losses = []
    val_losses = []

    best_vloss = float('inf')
    patience = 15
    pt_counter = 0

    date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f'date: {date}')

    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch + 1}")

        train_loss = train_loop(lmc_model, lm_train_loader, optimizer_lmc)
        train_losses.append(train_loss)

        val_loss = eval_loop(lmc_model, lm_test_loader)
        val_losses.append(val_loss)

        print(f"Training Loss: {train_loss}, Validation Loss {val_loss}")

        # Save metrics for future reference
        with open(f"../logs/model-metrics/model_metrics_lmc_{date}.json", 'w') as f:
            json.dump({"epoch": epoch, "training_loss": train_losses, "validation_loss": val_losses}, f)



        if val_loss < best_vloss:
            best_vloss = val_loss
            pt_counter = 0
            torch.save(lmc_model.state_dict(), f'../models/lmc_asl.pth')
        else:
            pt_counter += 1
            if pt_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def train_resnet18():
    """
    Trains the resnet18 model and saves a copy to storage for reuse
    """
    train_loader, test_loader = init_resnet18()

    train_losses = []
    val_losses = []

    date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f'date: {date}')
    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch + 1}")

        train_loss = train_loop(resnet18_model, train_loader, optimizer_resnet18)
        train_losses.append(train_loss)

        val_loss = eval_loop(resnet18_model, test_loader)
        val_losses.append(val_loss)

        # Save metrics for future reference
        with open(f"../logs/model-metrics/model_metrics_resnet18_{date}.json", 'w') as f:
            json.dump({"epoch": epoch, "training_loss": train_losses, "validation_loss": val_losses}, f)

        torch.save(resnet18_model.state_dict(), f'../models/resnet18_asl.pth')

        print(f"Training Loss: {train_loss}, Validation Loss {val_loss}")


if __name__ == '__main__':
    train_lmc() if MODEL_MODE == 'lmc' else train_resnet18()