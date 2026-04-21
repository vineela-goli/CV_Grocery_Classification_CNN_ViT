# Vineela Goli
# CS5330 - Computer Vision
# Final Project - Testing both models on real grocery photos

import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_data, get_transforms
from compare_models import load_resnet, load_vit

# Directory where real grocery photos are stored
PHOTO_DIR = 'data/real_photos'


def load_real_photos(photo_dir=PHOTO_DIR):
    """Load all images from the given directory and returns the images and filenames."""
    filenames = [f for f in os.listdir(photo_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    filenames.sort()
    images = []
    for fname in filenames:
        img = Image.open(os.path.join(photo_dir, fname)).convert('RGB')
        images.append(img)
    return images, filenames


def preprocess_image(image):
    """Loads the transform from file 1 and applies the test transform to the image passed in"""
    _, test_transform = get_transforms()
    return test_transform(image).unsqueeze(0)


def predict_single(network, image_tensor, class_names, is_resnet=True, device='cpu'):
    """Runs a single image tensor through the network and returns the predicted class name and confidence score."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = network(image_tensor)

    # ResNet outputs raw logits; ViT outputs log_softmax
    if is_resnet:
        probs = torch.softmax(output, dim=1)
    else:
        probs = torch.exp(output)

    confidence, pred_idx = probs.max(dim=1)
    return class_names[pred_idx.item()], confidence.item() * 100


def plot_real_results(images_pil, filenames, resnet_results, vit_results):
    """Display each real photo with the predictions from both models and the confidence score. Saves the output plot"""
    n = len(images_pil)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 5))
    for i, (img, fname) in enumerate(zip(images_pil, filenames)):
        r_class, r_conf = resnet_results[i]
        v_class, v_conf = vit_results[i]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        title = f"{fname}\nResNet: {r_class} ({r_conf:.1f}%)\nViT: {v_class} ({v_conf:.1f}%)"
        plt.title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig('real_photo_results.png')
    plt.show()


def main(argv):
    """Load both trained and saved models and run them on the real image set to evaluate model performance"""
    if not os.path.exists(PHOTO_DIR):
        print(f"'{PHOTO_DIR}' folder path is not found")
        return

    # load class names from the training dataset
    _, _, class_names = load_data()
    num_classes = len(class_names)

    print("Loading ResNet18...")
    resnet = load_resnet(num_classes)
    print("Loading ViT...")
    vit = load_vit(num_classes)

    images, filenames = load_real_photos()
    if not images:
        print(f"No images found in '{PHOTO_DIR}'")
        return

    print(f"\n{len(images)} images found. Running predictions...\n")
    print(f"{'Filename':<25} {'ResNet':<22} {'Conf':>6}   {'ViT':<22} {'Conf':>6}")
    print("-" * 90)

    resnet_results = []
    vit_results = []
    for img, fname in zip(images, filenames):
        tensor = preprocess_image(img)
        r_class, r_conf = predict_single(resnet, tensor, class_names, is_resnet=True)
        v_class, v_conf = predict_single(vit, tensor, class_names, is_resnet=False)
        resnet_results.append((r_class, r_conf))
        vit_results.append((v_class, v_conf))
        print(f"{fname:<25} {r_class:<22} {r_conf:>5.1f}%   {v_class:<22} {v_conf:>5.1f}%")

    plot_real_results(images, filenames, resnet_results, vit_results)

if __name__ == "__main__":
    main(sys.argv)
