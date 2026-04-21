# Vineela Goli
# CS5330 - Computer Vision
# Final Project - This file compares our trained ResNet18 and ViT performance on the fruits and 
# vegetables dataset

import sys
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from utils import load_data
from vit_model import GroceryViTConfig, NetTransformer


def load_resnet(num_classes, path='resnet18_fruits.pth'):
    """Load the saved ResNet18 model and return the model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_vit(num_classes, path='vit_fruits.pth'):
    """Loads the saved ViT model and return the model."""
    config = GroceryViTConfig(num_classes=num_classes)
    model = NetTransformer(config)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def get_predictions(network, images):
    """Runs the batch of images through the network passed in and returns predicted class indices."""
    with torch.no_grad():
        output = network(images)
    return output.argmax(dim=1).cpu().tolist()


def plot_comparison(images, labels, resnet_preds, vit_preds, class_names, num=8):
    """Plots a grid showing each image with its true label and both model predictions for comparison."""
    # undo normalization so images display correctly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    num = min(num, len(images))
    cols = 4
    rows = (num + cols - 1) // cols

    # Compute totals where both models are correct, both wrong and one correct
    both_correct = 0
    one_correct = 0
    both_wrong = 0

    plt.figure(figsize=(cols * 3, rows * 4))
    for i in range(num):
        img = images[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0)

        true_label = class_names[labels[i]]
        resnet_label = class_names[resnet_preds[i]]
        vit_label = class_names[vit_preds[i]]

        r_correct = (resnet_preds[i] == labels[i])
        v_correct = (vit_preds[i] == labels[i])

        if r_correct and v_correct:
            both_correct += 1
            color = 'green'
        elif r_correct or v_correct:
            one_correct += 1
            color = 'orange'
        else:
            both_wrong += 1
            color = 'red'

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        title = f"True: {true_label}\nResNet: {resnet_label}\nViT: {vit_label}"
        plt.title(title, fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

    print(f"\nBoth correct: {both_correct} | One correct: {one_correct} | Both wrong: {both_wrong}")


def main(argv):
    """Load both models and visualizes side-by-side predictions on test image set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader, class_names = load_data()
    num_classes = len(class_names)

    print("Loading ResNet18...")
    resnet = load_resnet(num_classes)
    print("Loading ViT...")
    vit = load_vit(num_classes)

    # grab one batch from the test set
    images, labels = next(iter(test_loader))
    labels_list = labels.tolist()

    resnet_preds = get_predictions(resnet, images)
    vit_preds = get_predictions(vit, images)

    plot_comparison(images, labels_list, resnet_preds, vit_preds, class_names, num=8)


if __name__ == "__main__":
    main(sys.argv)
