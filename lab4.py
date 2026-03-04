"""
CSCI 3485
Lab 4: Transfer Learning
Prathit Kurup & Victoria Figueroa
"""

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time

# Declare global variables for training and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
N_EPOCHS = 10
BATCH_SIZE = 64
NEURONS = 2048
LR = 1e-3
LOSS_FN = nn.CrossEntropyLoss()

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

@torch.no_grad()
def inference_time_per_image(model, model_name, sample_batch):
    """
    Measure one forward pass and divide by batch size.
    """
    model.eval()
    x, _ = sample_batch
    x = x.to(device)

    start = time.time()
    _ = model(x)
    elapsed = time.time() - start
    print(f"\nInference time per image for {model_name}: {elapsed / x.shape[0]:.6f} seconds")
    return elapsed / x.shape[0]

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

def run_experiment(model, model_weights, model_name):
    print(f"\nRunning experiment for {model_name}")

    # Create dataset and dataloaders using the transformations associated with the pre-trained model
    model_transformations = model_weights.transforms()
    train_dl, test_dl = create_dataset(model_transformations)

    # model = freeze_backbone(model)

    # Remove classifier depending on architecture
    if hasattr(model, "classifier"):    # VGG
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Identity()
    elif hasattr(model, "fc"):          # ResNet
        model.fc = nn.Identity()

    # Freeze backbone parameters to prevent training and save memory
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()

    # Run convlutional layers once to extract features using the pre-trained weights
    train_features, train_labels = run_conv_layers(train_dl, model)
    test_features, test_labels = run_conv_layers(test_dl, model)

    # Create dataloaders for the newly extracted features as the input to train the MLP classifier
    train_features_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_dl = DataLoader(train_features_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_features_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_dl = DataLoader(test_features_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build MLP classifier and train on the extracted features
    input_dim = train_features.shape[1]
    model = build_mlp_model(input_dim=input_dim, num_classes=10)
    n_losses, n_accuracies, n_train_time = train_model(model=model, model_name=model_name, train_dl=train_dl)

    # Visualize results
    visualize_training(losses=n_losses, accuracies=n_accuracies, model_name=model_name)
    print(f"Training time for {model_name}: {n_train_time:.2f} seconds")

    # Test classifier
    test_model(model=model, model_name=model_name, test_dl=test_dl)


def main():
    # VGG11
    vgg11_weights=models.VGG11_Weights.IMAGENET1K_V1
    vgg11_model = models.vgg11(weights=vgg11_weights)
    run_experiment(model=vgg11_model, model_weights=vgg11_weights, model_name="VGG11")

    # # VGG13
    # vgg13_weights=models.VGG13_Weights.IMAGENET1K_V1
    # vgg13_model = models.vgg13(weights=vgg13_weights)
    # run_experiment(model=vgg13_model, model_weights=vgg13_weights, model_name="VGG13")

    #  # ResNet18
    # resnet18_weights=models.ResNet18_Weights.IMAGENET1K_V1
    # resnet18_model = models.resnet18(weights=resnet18_weights)
    # run_experiment(model=resnet18_model, model_weights=resnet18_weights, model_name="ResNet18")

    #  # ResNet34
    # resnet34_weights=models.ResNet34_Weights.IMAGENET1K_V1
    # resnet34_model = models.resnet34(weights=resnet34_weights)
    # run_experiment(model=resnet34_model, model_weights=resnet34_weights, model_name="ResNet34")

if __name__ == "__main__":
    main()
