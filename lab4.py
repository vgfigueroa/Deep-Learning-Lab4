"""
CSCI 3485
Lab 4: Transfer Learning
Prathit Kurup & Victoria Figueroa
"""

from glob import glob
import torchvision.transforms as transforms
import torchvision.io as io
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time
import os

device = "mps" if torch.backends.mps.is_available() else 'cpu'
N_EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
LOSS_FN = nn.CrossEntropyLoss()

def freeze_original_weights(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def build_mlp_model(num_classes=10):
    model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(2048, num_classes)).to(device)
    summary(model.to("cpu"), (512,))
    model.to(device)
    return model

# Training, accuracy, testing, and eval copied for Lab 3
def train_batch(x, y, model, opt, loss_fn=LOSS_FN):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())

@torch.no_grad()
def accuracy_batch(x, y, model):
    model.eval()
    preds = model(x).argmax(dim=1)
    return float((preds == y).float().mean().cpu())

def train_model(model, model_name, train_dl, n_epochs=N_EPOCHS, loss_fn = LOSS_FN, lr=LR):
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    losses, accs = [], []
    start = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            epoch_losses.append(train_batch(x, y, model, opt, loss_fn))
        losses.append(float(np.mean(epoch_losses)))

        # quick train accuracy (structure; you can subsample later)
        epoch_accs = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            epoch_accs.append(accuracy_batch(x, y, model))
        accs.append(float(np.mean(epoch_accs)))

        print(f"epoch {epoch+1}/{n_epochs}  loss={losses[-1]:.4f}  train_acc={accs[-1]:.4f}")

    train_time = time.time() - start
    return losses, accs, train_time

@torch.no_grad()
def test_model(model, model_name,test_dl):
    accs = []
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        accs.append(accuracy_batch(x, y, model))
    return float(np.mean(accs))

@torch.no_grad()
def inference_time_per_image(model, sample_batch):
    """
    Optional metric suggested by lab: inference time.
    Measure one forward pass and divide by batch size.
    """
    model.eval()
    x, _ = sample_batch
    x = x.to(device)

    start = time.time()
    _ = model(x)
    elapsed = time.time() - start
    return elapsed / x.shape[0]

def main():
    # VGG16
    vgg16_weights=models.VGG16_Weights.IMAGENET1K_V1
    vgg16_model = models.vgg16(weights=vgg16_weights)
    print(vgg16_model)
    vgg16_model_transformations = vgg16_weights.transforms()
    
    train_ds = CIFAR10(root="./data", train=True, download=True, transform=vgg16_model_transformations)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = CIFAR10(root="./data", train=False, download=True, transform=vgg16_model_transformations)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    vgg16_model = freeze_original_weights(vgg16_model)
    vgg16_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    vgg16_model.classifier = build_mlp_model(num_classes=10)
    # summary(vgg16_model.to("cpu"), (3, 224, 224))

    # Train and test the model
    vgg16_model = vgg16_model.to(device)
    n_losses, n_accuracies,n_train_time = train_model(model=vgg16_model, model_name = "VGG16", train_dl=train_dl)
    print(f"Training time: {n_train_time:.2f} seconds")
    test_acc = test_model(model=vgg16_model, model_name="VGG16", test_dl=test_dl)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
