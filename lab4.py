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
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
device = "mps" if torch.backends.mps.is_available() else 'cpu'

n_epochs = 5
# I think this is how you set up the dataset, documentation was a little confusing
class CIFAR10Dataset(root = "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Deep-Learning-Lab4/cifar-100-python", train = True, download = True):
    def __init__(self, root, train):
        self.data = datasets.CIFAR100(root=root, train=train, download=True)
    def __getitem__(self, ix):
        x, y = self.data[ix]
        x = transforms.ToTensor()(x)
        return x.to(device), torch.tensor(y).to(device)
    def __len__(self):
        return len(self.data)


# todo
def data_loader():
    pass


# need to preprocess data before we give to pre-trained models
def apply_transformations(self):
    pass

def freeze_orginal_weights(model):
    for p in model.parameters():
        p.requires_grad = False

def build_mlp_model(in_features, num_classes=10):
    model = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)).to(device)
    summary(model.to("cpu"), (1, 28*28))
    model.to(device)
    return model

# here we fine tune?
def build_transfer_model(model,model_name, weights):
    freeze_orginal_weights(model)
    if model_name.startswith("vgg"):
        print("TODO")
    elif model_name.startswith("resnet"):
        print("TODO")





# the pretrained models
vgg_model16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg_model13 = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
resnet_model18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet_model34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)


# These are the transformations, that we would apply to data
vgg_model116_transformations = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).transforms()
vgg_model113_transformations = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1).transforms()
resnet_model118_transformations = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).transforms()
resnet_model134_transformations = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).transforms()


# From lab 3, we can reuse
def train_batch(x, y, model, opt, loss_fn):
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

def train_model(model, train_dl, n_epochs=n_epochs, lr=lr):
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
def test_model(model, test_dl):
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

    # need a data loader since 
    pass

if __name__ == "__main__":
    main()
