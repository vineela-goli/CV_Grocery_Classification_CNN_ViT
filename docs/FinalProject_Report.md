# Grocery Classification: CNN vs. Vision Transformer

**Vineela Goli**
CS 5330 Pattern Recognition and Computer Vision — Northeastern University

---

## Abstract

Identifying grocery products by category from a photograph is a practical problem with applications in meal planning, shopping assistance, and inventory management. This paper presents a comparative study of two image classification architectures applied to a 36-class fruit and vegetable dataset: a pretrained ResNet18 fine-tuned via transfer learning, and a Vision Transformer (ViT) trained from scratch. The ResNet18 model achieved 97.49% test accuracy after two-phase fine-tuning over 15 epochs, while the ViT reached 90.81% accuracy after 35 epochs. Both models were evaluated on held-out test images as well as real-world grocery photographs captured on a phone camera. Results confirm that transfer learning with a pretrained CNN offers a clear advantage on a moderately sized dataset, converging faster and with greater stability. The ViT, trained entirely from scratch, closes the gap considerably with extended training and represents a promising direction for future work with larger datasets or pretrained transformer weights.

---

## I. Introduction

Classifying food and grocery items from images is a deceptively difficult problem. Many categories share similar colors, shapes, and textures — corn and sweetcorn, for instance, or capsicum and bell pepper — making fine-grained visual distinctions necessary. At the same time, real-world images introduce lighting variation, background clutter, and viewpoint changes that controlled datasets do not capture. Solving this reliably has practical value across consumer applications, from smart grocery apps to automated checkout systems.

The dominant approach to image classification over the past decade has been the Convolutional Neural Network (CNN). CNNs exploit spatial locality through shared convolutional filters and have proven remarkably effective even when pretrained on large datasets such as ImageNet and fine-tuned on smaller domain-specific collections. More recently, Vision Transformers (ViTs) have emerged as a competitive alternative, applying the self-attention mechanism from natural language processing directly to sequences of image patches. ViTs achieve state-of-the-art results on large benchmarks but are known to require significantly more data or pretraining than CNNs to reach comparable performance on smaller datasets.

This project builds and compares these two approaches on a 36-class collection of fruit and vegetable images from Kaggle. The goal is to understand the practical tradeoffs between a well-established CNN baseline and a transformer architecture trained from scratch, both in terms of final accuracy and training dynamics. Both models are additionally tested on real photographs taken outside the training distribution to assess generalization.

---

## II. Related Work

Transfer learning with deep CNNs for food and produce classification has a well-established literature. He et al. introduced ResNet [1], demonstrating that residual connections allow very deep networks to be trained effectively. ResNet18 and its variants have since become standard baselines for transfer learning tasks due to their balance of depth and parameter efficiency. The pretrained ImageNet weights provide strong low-level feature representations that transfer well to grocery and food domains, making ResNet a natural first choice for this work.

Dosovitskiy et al. proposed the Vision Transformer [2], showing that a pure transformer applied to sequences of image patches can achieve competitive results on image classification benchmarks when trained on sufficiently large datasets. Their work directly defines the ViT architecture adapted in this project, including the patch embedding scheme and positional encoding. Critically, the authors note that ViTs underperform CNNs on smaller datasets when trained from scratch — a finding this project confirms empirically.

Martinel et al. [3] explored fine-grained food recognition across a large number of categories using CNN-based features combined with discriminative region pooling. Their findings reinforce that visually similar categories benefit from architectures that focus on local discriminative features rather than global statistics. This is directly relevant to the per-class accuracy analysis in this project, where categories such as apple, capsicum, and lettuce proved most challenging for the ViT.

---

## III. Methods

### A. Dataset

All experiments use the Grocery Store Dataset sourced from Kaggle [4]. The dataset contains images of 36 fruit and vegetable categories including apple, banana, beetroot, bell pepper, carrot, and watermelon, among others. The training split comprises 98 batches and the test split 12 batches at a batch size of 32.

**Preprocessing:**
- **Train:** Resize to 256 → RandomCrop to 224×224 → RandomHorizontalFlip → Normalize
- **Test:** Resize to 256 → CenterCrop to 224×224 → Normalize
- Normalization uses ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

### B. Model 1 — ResNet18 with Transfer Learning

The first model is a ResNet18 [1] loaded with pretrained ImageNet weights from torchvision. The final fully connected layer is replaced with a linear layer mapping 512 features to 36 classes. Training proceeds in two phases:

| Phase | Epochs | LR | Layers trained |
|-------|--------|----|----------------|
| 1 — Head only | 10 | 1e-3 | FC head only (base frozen) |
| 2 — Full fine-tune | 5 | 1e-4 | All layers unfrozen |

**Optimizer:** Adam | **Loss:** CrossEntropyLoss

### C. Model 2 — Vision Transformer from Scratch

The second model is a Vision Transformer adapted from the architecture described in [2]. Each 224×224 image is divided into 196 non-overlapping 16×16 patches using `nn.Unfold`, then projected into a 192-dimensional embedding space via a linear layer. Learnable positional embeddings are added before the sequence is passed through a stack of transformer encoder blocks.

| Hyperparameter | Value |
|----------------|-------|
| Patch size / stride | 16×16 (non-overlapping) |
| Number of tokens | 196 (14×14 grid) |
| Embedding dim | 192 |
| Encoder depth | 6 layers |
| Attention heads | 6 (32 dim/head) |
| Feedforward dim | 384 |
| Dropout | 0.1 |
| Pooling | Mean token pooling (no CLS token) |

After the encoder, tokens are averaged across the sequence dimension and passed through a two-layer classification head with GELU activation. The model is trained from scratch for 35 epochs using AdamW (lr=1e-3, weight decay=1e-4) with NLL loss and a cosine annealing learning rate schedule. An initial 20-epoch run was extended to 35 after the model had not yet converged.

**Optimizer:** AdamW | **Loss:** NLLLoss | **Scheduler:** CosineAnnealingLR

### D. Evaluation

Both models are evaluated on per-epoch test accuracy, final test accuracy, and per-class accuracy across all 36 categories. A side-by-side visual comparison is produced on a held-out test batch of 8 images. Both models are also tested on real-world grocery photographs taken on a phone camera and stored in a separate folder outside the training distribution.

---

## IV. Experiments and Results

### A. ResNet18 Training Dynamics

The two-phase training strategy proved highly effective. During Phase 1, test accuracy climbed from 79.67% at epoch 1 to 92.48% at epoch 10, with train loss dropping from 2.33 to 0.43. Phase 2 fine-tuning pushed accuracy to a final value of 97.49% by epoch 5, with train loss reaching 0.10. The loss curves (Figure 1) show clean, monotonic convergence with no sign of overfitting.

| Epoch | Train Loss | Test Accuracy |
|-------|-----------|---------------|
| 1 (Phase 1) | 2.3291 | 79.67% |
| 5 (Phase 1) | 0.6306 | 92.48% |
| 10 (Phase 1) | 0.4322 | 92.48% |
| 11 (Phase 2, FT1) | 0.4688 | 93.04% |
| 13 (Phase 2, FT3) | 0.1648 | 95.54% |
| **15 (Phase 2, FT5)** | **0.1028** | **97.49%** |

*Figure 1: Train/Test Loss Curves for ResNet18 Training*
![ResNet18 Training](../ResNet18_Training.png)

### B. ViT Training Dynamics

The ViT required substantially more epochs to converge. An initial 20-epoch run reached only 78.27% test accuracy. Extended training to 35 epochs improved this to 90.81%, with train loss falling from 2.96 to 0.69. The training curves (Figure 2) are noisier than ResNet's, with test accuracy fluctuating by several percentage points between consecutive epochs — consistent with the known instability of training ViTs from scratch on small datasets.

| Epoch | Train Loss | Test Accuracy |
|-------|-----------|---------------|
| 1 | 2.9616 | 27.30% |
| 10 | 1.7194 | 57.94% |
| 20 | 1.1078 | 78.55% |
| 28 | 0.7852 | 89.42% |
| **35** | **0.6895** | **90.81%** |

*Figure 2: Train/Test Loss Curves for ViT Training (35 Epochs)*
![ViT Training](../ViT_Training2.png)

### C. Per-Class Accuracy

ResNet18 achieved 100% accuracy on 29 of 36 classes. The seven classes below 100% were banana (77.8%), potato (80.0%), apple (90.0%), bell pepper (90.0%), carrot (90.0%), corn (90.0%), and sweetcorn (90.0%).

The ViT showed wider variance: 28 of 36 classes reached 90% or above, while apple (50.0%), lettuce (50.0%), potato (70.0%), and capsicum (80.0%) were the most problematic. Categories with strong and distinctive visual features — watermelon, pomegranate, soy beans — reached 100% on both models.

| Class | Correct | Total | Accuracy |
|-------|--------:|------:|---------:|
| apple | 5 | 10 | 50.0% |
| banana | 7 | 9 | 77.8% |
| beetroot | 10 | 10 | 100.0% |
| bell pepper | 9 | 10 | 90.0% |
| cabbage | 10 | 10 | 100.0% |
| capsicum | 8 | 10 | 80.0% |
| carrot | 10 | 10 | 100.0% |
| cauliflower | 10 | 10 | 100.0% |
| chilli pepper | 9 | 10 | 90.0% |
| corn | 9 | 10 | 90.0% |
| cucumber | 9 | 10 | 90.0% |
| eggplant | 9 | 10 | 90.0% |
| garlic | 10 | 10 | 100.0% |
| ginger | 9 | 10 | 90.0% |
| grapes | 10 | 10 | 100.0% |
| jalepeno | 8 | 10 | 80.0% |
| kiwi | 9 | 10 | 90.0% |
| lemon | 10 | 10 | 100.0% |
| lettuce | 5 | 10 | 50.0% |
| mango | 9 | 10 | 90.0% |
| onion | 10 | 10 | 100.0% |
| orange | 10 | 10 | 100.0% |
| paprika | 8 | 10 | 80.0% |
| pear | 10 | 10 | 100.0% |
| peas | 9 | 10 | 90.0% |
| pineapple | 10 | 10 | 100.0% |
| pomegranate | 10 | 10 | 100.0% |
| potato | 7 | 10 | 70.0% |
| raddish | 10 | 10 | 100.0% |
| soy beans | 10 | 10 | 100.0% |
| spinach | 9 | 10 | 90.0% |
| sweetcorn | 8 | 10 | 80.0% |
| sweetpotato | 10 | 10 | 100.0% |
| tomato | 10 | 10 | 100.0% |
| turnip | 10 | 10 | 100.0% |
| watermelon | 10 | 10 | 100.0% |

*Table 1: Per-Class Accuracy of ViT (35 Epochs)*

### D. Side-by-Side Comparison and Real-World Photos

On a random test batch of 8 images, both models agreed and were correct on 5, one model was correct on 2, and both were wrong on 1 (Figure 3).

| Outcome | Count |
|---------|------:|
| Both correct | 5 |
| One correct | 2 |
| Both wrong | 1 |

*Figure 3: Side-by-Side Model Comparison on Test Batch*
![Comparison](../comparison.png)

On five real-world phone camera photographs, ResNet18 correctly classified 4 of 5 while the ViT correctly classified 2 of 5 (Figure 4):

| Image | ResNet18 | Conf | ViT | Conf |
|-------|----------|-----:|-----|-----:|
| banana.jpeg | **banana** ✓ | 99.7% | soy beans ✗ | 94.9% |
| capsicum.jpeg | **capsicum** ✓ | 81.5% | banana ✗ | 40.6% |
| jalapeno.jpeg | **jalepeno** ✓ | 93.4% | **jalepeno** ✓ | 71.6% |
| onion.jpeg | **onion** ✓ | 67.4% | **onion** ✓ | 54.1% |
| potato.jpeg | **potato** ✓ | 36.6% | beetroot ✗ | 73.9% |

*Figure 4: Real-World Photo Prediction Results*
![Real Photos](../real_photo_results.png)

---

## V. Discussion and Summary

The results clearly favor the pretrained ResNet18. A 97.49% final accuracy versus 90.81% for the ViT is a meaningful gap, and ResNet achieved it in only 15 total epochs compared to 35 for the ViT. This aligns with the finding from Dosovitskiy et al. [2] that transformers require substantially more data or pretraining to match CNN performance — the grocery dataset is not large enough to fully exploit the ViT's capacity for learning global relationships.

The categories where both models struggled — apple, potato, capsicum — tend to be visually ambiguous or similar to related classes. Apple images varied across red and green varieties, potato shared texture with turnip and sweetpotato, and capsicum overlaps heavily with bell pepper. These cases likely require higher resolution, stronger augmentation, or fine-grained attention mechanisms to resolve reliably.

The real-world photo test exposed a gap between controlled test-set performance and true generalization. Both models were sensitive to background clutter and non-standard lighting, expected given the relatively simple training augmentation applied.

Future work could explore pretrained ViT weights (e.g., ViT-B/16 from ImageNet-21k) rather than training from scratch, which would likely close or reverse the accuracy gap. Stronger data augmentation such as mixup or cutmix could also improve robustness to real-world variation.

| Metric | ResNet18 | ViT (scratch) |
|--------|----------|---------------|
| Test accuracy | **97.49%** | 90.81% |
| Total epochs | 15 | 35 |
| Classes at 100% | 29/36 | 18/36 |
| Real-photo accuracy | **4/5** | 2/5 |
| Training approach | Transfer learning | From scratch |

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[3] N. Martinel, G. L. Foresti, and C. Micheloni, "Wide-Slice Residual Networks for Food Recognition," in *Proc. IEEE Winter Conf. Applications of Computer Vision (WACV)*, Santa Rosa, CA, USA, 2018, pp. 567–576.

[4] K. Seth, "Fruit and Vegetable Image Recognition Dataset," Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
