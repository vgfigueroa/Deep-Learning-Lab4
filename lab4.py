"""
CSCI 3485
Lab 4: Transfer Learning
Prathit Kurup & Victoria Figueroa
"""

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import datasets
import torch.nn as nn
from torchvision.io import read_image
from torchsummary import summary
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import time
import os

# Declare global variables for training and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
N_EPOCHS = 10
BATCH_SIZE = 64
NEURONS = 2048
LR = 1e-3
LOSS_FN = nn.CrossEntropyLoss()
google_paths = ["/content/gdrive/MyDrive/googleimages/cat/cat1.jpg",
  "/content/gdrive/MyDrive/googleimages/cat/cat2.jpg",
  "/content/gdrive/MyDrive/googleimages/dog/dog1.jpg",
  "/content/gdrive/MyDrive/googleimages/dog/dog2.jpg",
  "/content/gdrive/MyDrive/googleimages/frog/frog1.jpg",
  "/content/gdrive/MyDrive/googleimages/frog/frog2.jpg"]

# Training, accuracy, testing, and eval copied for Lab 3
def train_batch(x, y, model, opt, loss_fn):
    # Updated train batch function to return the loss and predictions for the batch

    model.train()                       # Set model to training mode
    opt.zero_grad()                     # Flush memory
    outputs = model(x)                  # Forward pass
    batch_loss = loss_fn(outputs, y)    # Compute loss
    batch_loss.backward()               # Compute gradients
    opt.step()                          # Make a GD step
    preds = outputs.argmax(dim=1)       # Get predicted class labels

    return batch_loss.detach().cpu(), preds.detach()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float())/len(y)
    return s.cpu().numpy()

def train_model(model, model_name, train_dl):
    loss_fn = LOSS_FN
    opt = Adam(model.parameters(), lr=LR)

    losses, accuracies = [], []
    start_time = time.time()
    print(f"\nBegin training for: {model_name}")
    for epoch in range(N_EPOCHS):
        print(f"{model_name}: epoch {epoch + 1} of {N_EPOCHS}")
        epoch_losses = []
        correct = 0
        total = 0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            batch_loss, preds = train_batch(x, y, model, opt, loss_fn)
            epoch_losses.append(batch_loss.item())
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = float(np.mean(epoch_losses))
        epoch_accuracy = correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"  loss={epoch_loss:.4f}, train_acc={epoch_accuracy:.4f}")

    end_time = time.time()
    train_time = end_time - start_time
    print("MLP training time (seconds): ", train_time)
    return losses, accuracies, train_time

@torch.no_grad()
def test_model(model, model_name,test_dl):
    accs = []
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        accs.append(accuracy(x, y, model))
    print(f"{model_name} test accuracy: {np.mean(accs)}")

def visualize_training(losses, accuracies, model_name):
    print(f"\nVisualizing training for {model_name}\n")
    plt.figure(figsize=(13,3))
    plt.subplot(121)
    plt.title(f'{model_name}: Training Loss')
    plt.plot(np.arange(N_EPOCHS) + 1, losses)
    plt.subplot(122)
    plt.title(f'{model_name}: Training Accuracy')
    plt.plot(np.arange(N_EPOCHS) + 1, accuracies)
    plt.show()

def build_mlp_model(input_dim, num_classes=10):
    '''Build a simple MLP model to replace the original classifier of the pre-trained model.'''
    model = nn.Sequential(
        nn.Linear(input_dim, NEURONS),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(NEURONS, num_classes)).to(device)
    # summary(model, (input_dim,), device="cpu")
    return model

@torch.no_grad()
def run_conv_layers(dl, model):
    print("Extracting features using frozen backbone...")

    start_time = time.time()
    model.eval()
    # Run convolutional layers once to extract features using the pre-trained weights
    features = []
    labels = []
    for x, y in dl:
        x = x.to(device)
        outputs = model(x)      # run CNN forward once
        outputs = torch.flatten(outputs, start_dim=1)
        features.append(outputs.detach().cpu())
        labels.append(y)

    end_time = time.time()
    extraction_time = end_time - start_time
    print("CNN feature extraction time (seconds): ", extraction_time)

    # This is the data we will train the MLP on
    return torch.cat(features), torch.cat(labels)

def freeze_backbone(model):
    # Freeze the backbone
    model.to(device)
    # model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def create_dataset(model_transformations):
    train_ds = CIFAR10(root="./data", train=True, download=True, transform=model_transformations)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_ds = CIFAR10(root="./data", train=False, download=True, transform=model_transformations)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dl, test_dl

def extract_vgg_features(vgg, x):
    feats = vgg.features(x)   # CNN feature extractor
    feats = torch.flatten(feats, 1)      # [B, C*H*W]
    return feats
def extract_resnet_features(resnet, x):
    feats = resnet.conv1(x)
    feats = resnet.bn1(feats)
    feats = resnet.relu(feats)
    feats = resnet.maxpool(feats)
    feats = resnet.layer1(feats)
    feats = resnet.layer2(feats)
    feats = resnet.layer3(feats)
    feats = resnet.layer4(feats)
    feats = resnet.avgpool(feats)
    feats = torch.flatten(feats, 1) # [B, C*H*W]
    return feats

@torch.no_grad()
def predict_and_show_google_images(backbone, mlp, weights, image_paths, model_name):
    backbone.eval().to(device)
    mlp.eval().to(device)

    class_names = datasets.CIFAR10(root="./data", train=False, download=True).classes
    preprocess = weights.transforms()

    for path in image_paths:
        img = read_image(path)  # [C,H,W], uint8 0..255

        # Convert 4-channel RGBA to 3-channel RGB if necessary
        if img.shape[0] == 4:
            img = img[:3, :, :]

        plt.figure(figsize=(3,3))
        plt.imshow(img.permute(1,2,0))
        plt.axis("off")

        # Preprocess for model
        x = preprocess(img)            # [C,224,224] normalized
        x = x.unsqueeze(0).to(device)  # [1,3,224,224]

        # Forward backbone + mlp
        feats = backbone(x)
        feats = torch.flatten(feats, 1)
        logits = mlp(feats)
        pred = logits.argmax(dim=1).item()

        plt.title(f"{model_name}\nPred: {class_names[pred]}\n{os.path.basename(path)}")
        plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def run_experiment(model, model_weights, model_name):
    print(f"\nRunning experiment for {model_name}")

    # Dataset using the correct preprocessing
    model_transformations = model_weights.transforms()
    train_dl, test_dl = create_dataset(model_transformations)

    # Remove classifier from backbone
    if hasattr(model, "classifier"):     # VGG
        model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.classifier = nn.Identity()

    elif hasattr(model, "fc"):           # ResNet
        model.fc = nn.Identity()

    backbone = model.to(device)

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    backbone.eval()

    train_features, train_labels = run_conv_layers(train_dl, backbone)
    test_features, test_labels = run_conv_layers(test_dl, backbone)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build MLP classifier
    input_dim = train_features.shape[1]
    mlp = build_mlp_model(input_dim=input_dim, num_classes=10).to(device)

    # Train classifier
    losses, accuracies, train_time = train_model(
        model=mlp,
        model_name=model_name,
        train_dl=train_dl
    )

    # Inference timing
    p = google_paths[0]
    img = read_image(p)
    x = model_weights.transforms()(img).unsqueeze(0).to(device)
    backbone.eval()
    mlp.eval()
    with torch.no_grad():
      if device == "cuda": torch.cuda.synchronize()
      t0 = time.time()
      feats = torch.flatten(backbone(x), 1); _ = mlp(feats)
    print(f"\nInference time per image for {model_name}: {(time.time()-t0):.6f} seconds")

    # Visualization
    visualize_training(losses, accuracies, model_name)
    print(f"Training time for {model_name}: {train_time:.2f} seconds")

    # Test accuracy
    test_model(mlp, model_name, test_dl)

    backbone_params = count_parameters(backbone)
    mlp_params = count_parameters(mlp)
    print(f"Total parameters in {model_name} backbone: {backbone_params}")
    print(f"Total parameters in {model_name} MLP: {mlp_params}")

    # return for google images test
    return backbone, mlp


def main():
    # VGG11
    vgg11_weights = models.VGG11_Weights.IMAGENET1K_V1
    vgg11_model = models.vgg11(weights=vgg11_weights)

    vgg11_backbone, vgg11_mlp = run_experiment(
        model=vgg11_model,
        model_weights=vgg11_weights,
        model_name="VGG11"
    )
    predict_and_show_google_images(vgg11_backbone, vgg11_mlp, vgg11_weights, google_paths, "VGG11")

    # VGG13
    vgg13_weights = models.VGG13_Weights.IMAGENET1K_V1
    vgg13_model = models.vgg13(weights=vgg13_weights)

    vgg13_backbone, vgg13_mlp = run_experiment(
        model=vgg13_model,
        model_weights=vgg13_weights,
        model_name="VGG13"
    )
    predict_and_show_google_images(vgg13_backbone, vgg13_mlp, vgg13_weights, google_paths, "VGG13")


    # ResNet18
    resnet18_weights = models.ResNet18_Weights.IMAGENET1K_V1
    resnet18_model = models.resnet18(weights=resnet18_weights)

    resnet18_backbone, resnet18_mlp = run_experiment(
        model=resnet18_model,
        model_weights=resnet18_weights,
        model_name="ResNet18"
    )
    predict_and_show_google_images(resnet18_backbone, resnet18_mlp, resnet18_weights, google_paths, "ResNet18")

    # ResNet34
    resnet34_weights = models.ResNet34_Weights.IMAGENET1K_V1
    resnet34_model = models.resnet34(weights=resnet34_weights)

    resnet34_backbone, resnet34_mlp = run_experiment(
        model=resnet34_model,
        model_weights=resnet34_weights,
        model_name="ResNet34"
    )
    predict_and_show_google_images(resnet34_backbone, resnet34_mlp, resnet34_weights, google_paths, "ResNet34")


if __name__ == "__main__":
  main()