# Vineela Goli
# CS5330 - Computer Vision
# Final Project - Builds and trains the ResNet18 transfer learning model. 
# This loads a pretrained ResNet18, freezes the base layers, replaced the final classification head
# for the 36 grocery classes, then trains head only first, then the full tune up. The model is also saved
# as resnet18_fruits.pth

import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from FinalProject_1 import load_data, train_network, test_network, plot_loss_curves, save_model, evaluate_per_class

def build_resnet(num_classes):
    """Load a pretrained ResNet18 model, freeze all layers but just replace
    the FC layer for the number of grocery classes."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # freeze all pretrained layers so we only train the new head
    for param in model.parameters():
        param.requires_grad = False

    # replace classification head for the number of grocery classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def main(argv):
    """Load data, build the ResNet model, run phase 1 training with base layers frozen and then phase 2 with 
    all layers unfrozen, plot the train/test loss curves, evaluate the accuracy per class and save the trained network"""
    
    # Adding here since my local PC is CPU and very slow, running the network training on Google CoLab to utilize 
    # free GPU and speed up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data sets and build ResNet network
    train_loader, test_loader, class_names = load_data()
    network = build_resnet(num_classes=len(class_names))
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []

    # Phase 1: train only the classification head with the base frozen
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=1e-3)
    print("\nPhase 1: Training classification head (base layers frozen)")
    for epoch in range(10):
        train_loss = train_network(network, train_loader, optimizer, criterion, device)
        test_loss, accuracy = test_network(network, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Accuracy={accuracy:.2f}%")

    # Phase 2: unfreeze all layers and fine-tune the model
    print("\nPhase 2: Fine-tuning full network")
    for param in network.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    for epoch in range(5):
        train_loss = train_network(network, train_loader, optimizer, criterion, device)
        test_loss, accuracy = test_network(network, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Fine-tune Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Accuracy={accuracy:.2f}%")

    plot_loss_curves(train_losses, test_losses, title='ResNet18 Training')
    evaluate_per_class(network, test_loader, class_names, device)
    save_model(network, "resnet18_fruits.pth")

if __name__ == "__main__":
    main(sys.argv)
