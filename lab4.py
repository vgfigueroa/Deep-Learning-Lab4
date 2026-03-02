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

# Declare global variables for training and evaluation
device = "mps" if torch.backends.mps.is_available() else 'cpu'
N_EPOCHS = 10
BATCH_SIZE = 64
NEURONS = 1024
LR = 1e-3
LOSS_FN = nn.CrossEntropyLoss()

def freeze_original_weights(model):
    '''Freeze the weights of the original model to prevent them from being updated during training.'''
    for p in model.parameters():
        p.requires_grad = False
    return model

def build_mlp_model(num_classes=10):
    '''Build a simple MLP model to replace the original classifier of the pre-trained model.'''
    model = nn.Sequential(
        nn.Linear(512, NEURONS),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(NEURONS, num_classes)).to(device)
    # summary(model.to("cpu"), (512,))
    model.to(device)
    return model

# Training, accuracy, testing, and eval copied for Lab 3
def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad() # Flush memory
    batch_loss = loss_fn(model(x), y) # Compute loss
    batch_loss.backward() # Compute gradients
    opt.step() # Make a GD step
    return batch_loss.detach().cpu()
    
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

        # Train and track train loss
        epoch_losses = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            batch_loss = train_batch(x, y, model, opt, loss_fn)
            epoch_losses.append(batch_loss)
        epoch_loss = float(np.mean(epoch_losses))

        # Track train accuracy
        epoch_accuracies = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        epoch_accuracy = float(np.mean(epoch_accuracies))

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"  loss={epoch_loss:.4f}, train_acc={epoch_accuracy:.4f}")

    end_time = time.time()
    train_time = end_time - start_time
    print("Training time (seconds): ", train_time)
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

def visualize_training_time(train_times, model_names, experiment_name):
    print(f"\nVisualizing training time for {experiment_name}")
    plt.figure(figsize=(8,5))
    plt.title(f'{experiment_name} Training Time (per model)')
    plt.bar(model_names, train_times)
    plt.ylabel('Time (seconds)')
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

def transfer_learning(model):
    model = freeze_original_weights(model)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = build_mlp_model(num_classes=10)
    summary(model.to("cpu"), (3, 224, 224))
    return model

def create_dataset(model_transformations):
    train_ds = CIFAR10(root="./data", train=True, download=True, transform=model_transformations)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = CIFAR10(root="./data", train=False, download=True, transform=model_transformations)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, test_dl

def run_experiment(model, model_weights, model_name):
    print(f"\nRunning experiment for {model_name}\n")

    # Create dataset and dataloaders using the transformations associated with the pre-trained model
    model_transformations = model_weights.transforms()
    train_dl, test_dl = create_dataset(model_transformations)

    # Perform transfer learning and train the model
    model = transfer_learning(model)
    model = model.to(device)
    n_losses, n_accuracies, n_train_time = train_model(model=model, model_name=model_name, train_dl=train_dl)

    # Visualize training and evaluation results
    visualize_training(losses=n_losses, accuracies=n_accuracies, model_name=model_name)
    visualize_training_time(train_times=[n_train_time], model_names=[model_name], experiment_name="Transfer Learning")
    # inference_time_per_image = inference_time_per_image(model=model, model_name=model_name, sample_batch=next(iter(test_dl)))
    test_model(model=model, model_name=model_name, test_dl=test_dl)


def main():
    # VGG16
    vgg16_weights=models.VGG16_Weights.IMAGENET1K_V1
    vgg16_model = models.vgg16(weights=vgg16_weights)
    run_experiment(model=vgg16_model, model_weights=vgg16_weights, model_name="VGG16")

    # # VGG19
    # vgg19_weights=models.VGG19_Weights.IMAGENET1K_V1
    # vgg19_model = models.vgg19(weights=vgg19_weights)
    # run_experiment(model=vgg19_model, model_weights=vgg19_weights, model_name="VGG19")

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
