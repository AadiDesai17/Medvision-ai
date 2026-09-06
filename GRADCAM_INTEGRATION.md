# MedVision — Grad-CAM Integration Guide
**Author:** K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)  
**Target Recipient:** Aadi (ROLE: Explainable AI / Grad-CAM Feature Owner)  
**Module:** `medvision_disease_prediction`  
**Model Architecture:** Pretrained `ResNet50` Fine-Tuned Classifier  

---

## Overview

Hi Aadi! This document explains exactly how to integrate your Grad-CAM module with my trained Disease Prediction model without needing to know any of my training or dataset preparation internals.

My model class (`MedVisionResNet50`) exposes a dedicated helper method `get_target_layer()` designed specifically for activation mapping.

---

## 1. Quick Integration Workflow

```
Input Image ──> Preprocessing ──> Forward Pass ──> Predicted Class Logit
                                        │                      │
                                        ▼                      ▼
                            [Target Conv Layer] ──> Backward Pass
                                        │
                                        ▼
                                  Grad-CAM Heatmap
```

---

## 2. Step-by-Step Code Example for Aadi

Here is a self-contained code snippet showing how to load my model, obtain the target layer, hook forward activations and backward gradients, and compute the Grad-CAM heatmap:

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# 1. Import preprocessor and model loader from Saaketh's module
from src.preprocessing import preprocess_image, load_image_rgb
from src.model import load_trained_model
from src.config import DEVICE, CLASS_NAMES

# 2. Load the trained model
model = load_trained_model(device=DEVICE)
model.eval()

# 3. Access the target convolutional layer for Grad-CAM
# This returns: model.backbone.layer4[-1] (the final Bottleneck block)
target_layer = model.get_target_layer()

# 4. Storage for activations and gradients
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Attach hooks to the target layer
f_handle = target_layer.register_forward_hook(forward_hook)
b_handle = target_layer.register_full_backward_hook(backward_hook)

# 5. Preprocess the input medical image
image_path = "path/to/chest_xray.png"
input_tensor = preprocess_image(image_path, device=DEVICE)  # Shape: [1, 3, 224, 224]

# 6. Forward Pass
logits = model(input_tensor)
probs = F.softmax(logits, dim=1)
predicted_idx = torch.argmax(probs, dim=1).item()
predicted_class = CLASS_NAMES[predicted_idx]
confidence = probs[0, predicted_idx].item()

print(f"Predicted Class: {predicted_class} ({confidence*100:.2f}%)")

# 7. Backward Pass for the target class score
model.zero_grad()
target_score = logits[0, predicted_idx]
target_score.backward()

# 8. Compute Grad-CAM
# Extract gradients & feature maps from hooks
grads = gradients[0].cpu().data.numpy()[0]          # Shape: [C, H, W] = [2048, 7, 7]
acts = activations[0].cpu().data.numpy()[0]          # Shape: [C, H, W] = [2048, 7, 7]

# Global Average Pooling on gradients to get importance weights
weights = np.mean(grads, axis=(1, 2))                # Shape: [2048]

# Weighted combination of forward activation maps
cam = np.zeros(acts.shape[1:], dtype=np.float32)     # Shape: [7, 7]
for i, w in enumerate(weights):
    cam += w * acts[i, :, :]

# Pass through ReLU (only positive contributions to the class matter)
cam = np.maximum(cam, 0)

# Normalize heatmap to [0, 1]
if cam.max() > 0:
    cam = cam / cam.max()

# Resize heatmap to match the original 224x224 image dimensions
cam_resized = cv2.resize(cam, (224, 224))

# 9. Clean up hooks
f_handle.remove()
b_handle.remove()

# cam_resized is now ready to be colormapped (cv2.COLORMAP_JET) and overlaid onto the original radiograph!
```

---

## 3. Specifications Summary for Aadi

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | `ResNet50` | Pretrained torchvision backbone |
| **Input Shape** | `[1, 3, 224, 224]` | Float32 RGB tensor |
| **Target Layer** | `model.get_target_layer()` | Specifically `backbone.layer4[-1]` |
| **Feature Map Dimensions** | `[2048, 7, 7]` | Channels=2048, Spatial resolution=7x7 |
| **Output Logits** | Shape `[1, 2]` | Index `0` = Normal, Index `1` = Pneumonia |
| **Loss / Score to Backprop** | `logits[0, target_class_idx]` | Backpropagate raw scalar logit (do NOT backprop softmax) |

---

## 4. Contact & Support
If you need any custom hooks, layer adjustments, or return formats, ping **K. Gnana Saaketh**.
