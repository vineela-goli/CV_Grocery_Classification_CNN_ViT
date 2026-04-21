# Vineela Goli
# CS5330 - Computer Vision
# Final Project - Grocery Classification: CNN vs. Vision Transformer
# This code file acts as a shared file for all remaining tasks. It handles data loading,
# transforms, training/testing loops, and plotting so every other file can just import it from
# here rather than having duplicate code per model. 

import sys
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Location of our training and test data set
DATA_DIR = './data/fruits_vegetables'
BATCH_SIZE = 32
IMAGE_SIZE = 224

def get_transforms():
    """Image preprocessing pipelines are defined for both test and train sets. 
    Images are augmented for training with random crop and flip and clean center 
    crop for testing. Both sets are normalized with mean and std."""
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def load_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """Loads the train and test datasets, transforms using get_transforms(),
    converts to dataLoader and returns the dataLoaders and class names."""
    train_transform, test_transform = get_transforms()

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names


def train_network(network, train_loader, optimizer, criterion, device='cpu'):
    """The shell train network function that can be used from all our other code files to train 
    one epoch with the network passed in. Returns the average training loss"""
    network.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

def test_network(network, test_loader, criterion, device='cpu'):
    """The shell test network function that can be used from all our other code files to evaluate
    a network with no_grad, compute average loss and accuracy and return them"""
    network.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = network(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_loss += loss.item()

    return (total_loss / len(test_loader)), (100 * correct / len(test_loader.dataset))

def plot_loss_curves(train_losses, test_losses, title='Training Curves'):
    """This is a helper function to plot training and test loss across all epochs in a single
    plot. The plot is also saved as a .png file"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, color='blue', label='Train Loss')
    plt.plot(epochs, test_losses, color='green', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_') + '.png')
    plt.show()
    
def save_model(network, path):
    """Saves the model state dict to the given path."""
    torch.save(network.state_dict(), path)
    print(f"Model saved to {path}")
    
def evaluate_per_class(network, test_loader, class_names, device='cpu'):
    """Loops through the test set, tracks correct/total per class and prints the results
    in a formatted accuracy table"""
    
    correct = {c: 0 for c in range(len(class_names))}
    total = {c: 0 for c in range(len(class_names))}

    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            preds = output.argmax(dim=1)
            for label, pred in zip(target, preds):
                total[label.item()] += 1
                if label == pred:
                    correct[label.item()] += 1

    print("\nPer-class accuracy:")
    print(f"{'Class':<25} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 52)
    per_class_accy = {}
    for i, name in enumerate(class_names):
        acc = 100 * correct[i] / total[i] if total[i] > 0 else 0
        per_class_accy[name] = acc
        print(f"{name:<25} {correct[i]:>8} {total[i]:>6} {acc:>7.1f}%")

    return per_class_accy

def show_sample_images(data_loader, class_names, num=8):
    """Display a grid of sample images from the dataset with category labels."""
    images, labels = next(iter(data_loader))

    plt.figure(figsize=(12, 6))
    for i in range(min(num, len(images))):
        img = images[i]
        plt.subplot(2, num // 2, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(class_names[labels[i].item()], fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(argv):
    """Load the train and test data, print the dataset metrics
    and show sample images"""
    train_loader, test_loader, class_names = load_data()

    print(f"Number of classes: {len(class_names)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"\nClasses: {class_names}")

    show_sample_images(train_loader, class_names)

if __name__ == "__main__":
    main(sys.argv)
