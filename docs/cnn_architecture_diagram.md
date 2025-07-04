# CNN Architecture: Base and Head Layers

## General CNN Architecture Diagram

```
INPUT IMAGE (e.g., 224×224×3)
         |
         ▼
╔═══════════════════════════════════════════════════════════════╗
║                          BASE LAYERS                          ║
║                    (Feature Extraction)                       ║
╠═══════════════════════════════════════════════════════════════╣
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │              CONVOLUTIONAL BLOCKS                       │  ║
║  │                                                         │  ║
║  │  Conv2D → BatchNorm → ReLU → MaxPool                   │  ║
║  │       ↓                                                 │  ║
║  │  Conv2D → BatchNorm → ReLU → MaxPool                   │  ║
║  │       ↓                                                 │  ║
║  │  Conv2D → BatchNorm → ReLU → MaxPool                   │  ║
║  │       ↓                                                 │  ║
║  │  Conv2D → BatchNorm → ReLU → MaxPool                   │  ║
║  │       ↓                                                 │  ║
║  │  Conv2D → BatchNorm → ReLU → MaxPool                   │  ║
║  └─────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════╝
         |
         ▼ (Feature Maps: e.g., 7×7×512)
╔═══════════════════════════════════════════════════════════════╗
║                          HEAD LAYERS                          ║
║                    (Task-Specific Output)                     ║
╠═══════════════════════════════════════════════════════════════╣
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │                 GLOBAL POOLING                          │  ║
║  │           (7×7×512 → 1×1×512)                          │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                           |                                   ║
║                           ▼                                   ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │                    FLATTEN                              │  ║
║  │                  (512 features)                         │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                           |                                   ║
║                           ▼                                   ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │               FULLY CONNECTED                           │  ║
║  │                Dense(256) → ReLU                        │  ║
║  │                     ↓                                   │  ║
║  │                 Dropout(0.5)                            │  ║
║  │                     ↓                                   │  ║
║  │              Dense(num_classes)                         │  ║
║  │                     ↓                                   │  ║
║  │                  Softmax                                │  ║
║  └─────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════╝
         |
         ▼
    OUTPUT PREDICTIONS
```

---

## Layer-by-Layer Breakdown

### BASE LAYERS (Feature Extraction Backbone)
```
Purpose: Extract hierarchical features from raw images
Characteristics:
• Convolutional layers with increasing depth
• Spatial dimensions decrease, feature depth increases
• Learn general visual patterns (edges → textures → objects)
• Often pre-trained on large datasets (ImageNet)
• Transferable across different tasks
```

### HEAD LAYERS (Task-Specific Classifier)
```
Purpose: Map extracted features to specific task outputs
Characteristics:
• Convert 2D feature maps to 1D vectors
• Dense layers for final classification/regression
• Task-specific (classification, detection, segmentation)
• Usually trained from scratch for new tasks
• Smaller parameter count compared to base
```

---

## Concrete Example: ResNet-50 for Image Classification

### INPUT
```
Image: 224×224×3 (RGB photo of a cat)
```

### BASE LAYERS (ResNet-50 Backbone)
```
Layer 1:  Conv2D(7×7, 64)     → 112×112×64
          MaxPool(3×3)        → 56×56×64

Layer 2:  ResBlock × 3        → 56×56×256
          [Conv2D(1×1,64) → Conv2D(3×3,64) → Conv2D(1×1,256)]

Layer 3:  ResBlock × 4        → 28×28×512
          [Conv2D(1×1,128) → Conv2D(3×3,128) → Conv2D(1×1,512)]

Layer 4:  ResBlock × 6        → 14×14×1024
          [Conv2D(1×1,256) → Conv2D(3×3,256) → Conv2D(1×1,1024)]

Layer 5:  ResBlock × 3        → 7×7×2048
          [Conv2D(1×1,512) → Conv2D(3×3,512) → Conv2D(1×1,2048)]
```

### HEAD LAYERS (Classification Head)
```
Global Avg Pool:  7×7×2048    → 1×1×2048
Flatten:          1×1×2048    → 2048
Dense:            2048        → 1000 (ImageNet classes)
Softmax:          1000        → [0.001, 0.85, 0.002, ...] (probabilities)
```

### OUTPUT
```
Prediction: "Cat" (class 281) with 85% confidence
```

---

## Feature Evolution Through Layers

```
INPUT IMAGE          BASE LAYER 1         BASE LAYER 2         BASE LAYER 3         HEAD LAYERS
     📷         →        ╱╲╱╲          →      ⚬ ⚬ ⚬        →      🐱 🐕 🐦      →    [Cat: 85%]
(224×224×3)           (Edges/Lines)        (Textures)         (Object Parts)        [Dog: 10%]
                      (112×112×64)        (56×56×256)        (28×28×512)          [Bird: 5%]
```

---

## Transfer Learning Example

### Pre-trained Base (Frozen)
```
ResNet-50 Base → Trained on ImageNet (1M+ images, 1000 classes)
Features learned: edges, textures, shapes, basic objects
```

### New Task Head (Trainable)
```
Medical X-ray Classification (3 classes: Normal, Pneumonia, COVID)

HEAD LAYERS:
Global Avg Pool:  7×7×2048 → 2048
Dense:           2048 → 512 (ReLU + Dropout)
Dense:           512 → 128 (ReLU + Dropout)  
Dense:           128 → 3 (Softmax)

OUTPUT: [Normal: 15%, Pneumonia: 80%, COVID: 5%]
```

This architecture allows the model to leverage powerful pre-trained features while adapting to new specific tasks efficiently!