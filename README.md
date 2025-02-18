# CV-StyleTransfer

The `project_cv.py` script represents a comprehensive implementation of foundational concepts in **computer vision**, structured to provide both educational insights and practical applications. this notebook combines the implementation of convolutional neural networks (CNNs) from scratch with the use of pre-trained VGG19 for **neural style transfer** tasks. The project is suited for learners and enthusiasts exploring topics in computer vision and machine learning.

## Key Highlights
1. **Hands-on Implementation of Convolutional Layers from Scratch**:
   - Learn how convolution operations work by manually implementing them in PyTorch.
   - Understand forward and backward propagation in CNNs visually and interactively.
   - Test and validate custom convolution operations against known techniques.

2. **Neural Style Transfer with Pre-trained VGG19**:
   - Blend the artistic style of one image with the content of another using Neural Style Transfer.
   - Visualize learned features from deep layers of convolutional networks.
   - Experiment with content and style tuning using advanced loss functions like Gram Matrices.

---

## File Overview 

The file is logically divided into two key sections detailed below:

### **1. Convolutional Neural Network from Scratch**

This section walks through implementing and testing convolutions manually, before relying on libraries like PyTorch.

- **Custom Convolution Class**:
    - Implements a `MyConv` class with:
      - **Forward Pass**: Processing input tensors with custom convolution filters.
      - **Backward Pass**: Computing gradients for both weights (filters) and input, essential for training CNN models.
    - Features stride, padding-handling, and kernel dimension functionality.

- **Testing Forward & Backward Convolution**:
    - Integration with custom unit tests validates output dimensions, numerical stability, and correctness.
    - Demonstrations of practical use with custom kernels like:
        - **Sobel Edge Detectors** for edge highlighting.
        - **Gaussian Kernels** for smooth blurring.

- **Visualization**:
    - Sample outputs of convolution filters on test images using Matplotlib.

---

### **2. Neural Style Transfer with VGG19**

A practical implementation of **Neural Style Transfer**, using the pre-trained **VGG19** CNN model as the backbone.

#### Key Steps:

1. **Preprocessing Content & Style Images**:
   - Loading and transforming images (e.g., resizing, cropping, normalization, and tensor conversion) to prepare for model inputs.

2. **Using VGG19 for Feature Extraction**:
   - Extract style and content information from intermediate convolution layers of the VGG-19 network.
   - Features extracted at different layers represent either:
     - **High-level structure and content** (early layers).
     - **Artistic or texture-like style** (deeper layers).

3. **Loss Functions**:
   - Implementing advanced loss trade-offs between style fidelity and content replication:
     - **Content Loss**: Ensures the generated image retains structural similarity with the base image.
     - **Style Loss**: Captures artistic elements of the style image using **Gram Matrices**.

4. **Optimization**:
   - Style transfer optimization is carried out using the **LBFGS optimizer**, tweaking the input image itself to minimize total error.
   - The total loss combines both style and content constraints, adjusting weights dynamically for aesthetic outcomes.

5. **Intermediate & Final Visualization**:
   - Display intermediate transfers during training to monitor convergence.
   - Create plots for metrics over iterations (*style loss, content loss, and total optimization loss*).

---

## Requirements
To run this project, make sure the following Python libraries are installed:
- `torch` — PyTorch library for deep learning and tensor operations.
- `torchsummary` — A layer-by-layer architecture summary visualization tool.
- `torchvision` — For pre-trained CNN models and image utilities.
- `Pillow (PIL)` — Image-loading and manipulation in Python.
- `matplotlib` — Standard library for plotting visualizations.
- `numpy` — Fundamental library for numerical computations.

Package installation command:
```bash
pip install torch torchvision matplotlib pillow numpy
```

---

## Outcomes

### 1. **Custom CNN Results**:
Convolution operations are demonstrated with:
- Edge-detection operators (like Sobel filters) successfully highlighting boundaries.
- Gaussian filtering for general blurring effects.

### 2. **Style Transfer Results**:
- The resulting image combines the formal content of the "content image" with the artistic visual stylization of the "style image" (`Neural Style Transfer`).

**Visual Examples**:
- Before optimization: Raw initialization images.
- After several iterations: Progressively refined style transfer.

---

## Concluding Remarks
This project strikes a balance between theoretical depth (*manual convolution examples*) and practical applications (*neural style transfer*). It’s ideal for:
- Students learning CNN internals firsthand.
- Researchers exploring applications of deep vision models like VGG19.
- Artists or engineers experimenting with AI-powered image stylization tools.
