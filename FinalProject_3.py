# Vineela Goli
# CS5330 - Computer Vision
# Final Project - Build a Vision Transformer from scratch. This is adapted from Project 5 Task 4. 

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from FinalProject_1 import load_data, train_network, test_network, plot_loss_curves, save_model, evaluate_per_class


class GroceryViTConfig:
    """Configuration for the ViT. Adapted from Project 5 NetConfig for 224x224 RGB images."""

    def __init__(self, num_classes):
        # dataset fixed attributes
        self.image_size = 224
        self.in_channels = 3
        self.num_classes = num_classes

        # 16x16 non-overlapping patches on a 224x224 image gives 14x14 = 196 tokens
        self.patch_size = 16
        self.stride = 16

        # transformer settings
        self.embed_dim = 192
        self.depth = 6
        self.num_heads = 6       # 192 / 6 = 32 per head
        self.mlp_dim = 384
        self.dropout = 0.1
        self.use_cls_token = False

        # training settings
        self.epochs = 35
        self.batch_size = 32
        self.lr = 1e-3
        self.weight_decay = 1e-4


class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Input:
        x of shape (B, C, H, W)

    Output:
        tokens of shape (B, N, D)

    where:
        B = batch size
        N = number of patches (tokens)
        D = embedding dimension
    """

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # non-overlapping patches when stride == patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # each extracted patch is flattened into one vector
        self.patch_dim = in_channels * patch_size * patch_size

        # After flattening a patch, project it into embedding space.
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced for this image setup
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        """
        Compute how many patches are extracted in total.
        Number of positions along one spatial dimension:
        ((image_size - patch_size) // stride) + 1
        Since the image is square and the patch is square, total patches are:
        positions_per_dim * positions_per_dim
        """
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        
        return positions_per_dim * positions_per_dim

    def forward(self, x):
        """
        Extract patches and convert them to embeddings.

        Input:
            x shape = (B, C, H, W)

        Output:
            x shape = (B, N, D)
        """
        # Step 1: extract patches using nn.Unfold, the shape becomes (B, patch_dim, N)
        x = self.unfold(x)
        # Step 2: move dimensions so each patch becomes one row/token.
        x = x.transpose(1, 2)
        # Step 3: project each flattened patch into embedding space.
        x = self.proj(x)
        
        return x

# The Transformer Network class
#
# network structure
#
# Patch embedding layer
# dropout
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Token averaging
# Linear layer w/GELU and dropout
# Fully connected output layer 10 nodes: softmax output
class NetTransformer(nn.Module):
    # the init method defines the layers of the network
    def __init__(self, config):
        
        # create all of the layers that have to store information
        super(NetTransformer, self).__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens))

        self.use_cls_token = config.use_cls_token

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Use the Torch Transformer Encoder Layer
        # transformer layer includes
        # multi-head self attention
        # feedforward network
        # layer normalization
        # residual connections
        # dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

        return

    def forward(self, x):
        """Runs a forward pass through the transformer.
        Takes an input image batch of shape (B, 3, 224, 224) and returns
        class log-probabilities of shape (B, num_classes)."""
        # execute the patch embedding layer
        x = self.patch_embed(x)
        # get the batch size (0 dimension of x)
        batch_size = x.size(0)

        # add the optional CLS token to the set 
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # add learnable positional embedding to each token
        x += self.pos_embed
        # run the dropout layer right after the patch embedding
        x = self.pos_dropout(x)
        # run the transformer encoder
        x = self.encoder(x)

        # pool tokens - either use CLS token or average all tokens
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # final normalization of the token to classify
        x = self.norm(x)
        x = self.classifier(x)
        # return the softmax of the output layer
        return F.log_softmax(x, dim=1)

def main(argv):
    """Builds ViT from scratch, trains it, plots train/test loss curves, evaluates per class, and saves model."""
    
    # Adding here since my local PC is CPU and very slow, running the network training on Google CoLab to utilize 
    # free GPU and speed up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data sets and build ViT network
    train_loader, test_loader, class_names = load_data()

    # Create our network and run the network on 35 epoch on the data (initially tried 20 but that wasn't enough), track train/test losses and accuracies and plot them
    config = GroceryViTConfig(num_classes=len(class_names))
    network = NetTransformer(config)
    network = network.to(device)

    # AdamW is standard for ViT training; NLLLoss matches the log_softmax output
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.NLLLoss()
    
    # Adjust learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        train_loss = train_network(network, train_loader, optimizer, criterion, device)
        test_loss, accuracy = test_network(network, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Accuracy={accuracy:.2f}%")

    plot_loss_curves(train_losses, test_losses, title='ViT Training')
    evaluate_per_class(network, test_loader, class_names, device)
    save_model(network, 'vit_fruits.pth')


if __name__ == "__main__":
    main(sys.argv)
