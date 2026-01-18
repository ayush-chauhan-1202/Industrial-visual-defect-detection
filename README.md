# Industrial-visual-defect-detection
Unsupervised industrial anomaly detection using a U-Net autoencoder on MVTec (PyTorch, MPS/CUDA support, AUROC 0.75)

# Problem Statement
Industrial inspection systems frequently suffer from a lack of labeled defect data. In real production environments, defective samples are rare, diverse, and expensive to annotate, while normal samples are abundant. This project addresses the core challenge of:

Detecting subtle defects in industrial images using only normal (defect-free) training data.

This is a foundational problem in industrial vision, quality assurance, and non-destructive evaluation (NDE).


# Motivation
This project is designed as a portfolio-grade demonstration for roles involving:

- Industrial computer vision

- Applied machine learning in manufacturing

- Visual inspection systems

- Anomaly detection in safety-critical domains

The emphasis is on:

- Realistic constraints (no defect labels during training)

- Clear problem framing

- Reproducible experiments

- Interpretable outputs (heatmaps)

- Clean engineering and code structure


# Data
We use the MVTec Anomaly Detection dataset, a widely adopted benchmark for industrial anomaly detection.

- Category used: tile

- Training data: Only good samples (unsupervised setup)

- Test data: Mix of good + anomalous samples

- Ground-truth anomaly masks used only for evaluation


Expected directory structure:

```
data/mvtec/tile/
├── train/good/
├── test/good/
├── test/crack/
├── test/glue_strip/
├── test/oil/
└── ground_truth/
```
Examples from **tile** class data: 

good image
<img width="200" height="200" alt="008" src="https://github.com/user-attachments/assets/a3fbca8b-529d-4781-a948-f4d7eb439ae2" />
Anomaly image (Crack)
<img width="200" height="200" alt="005" src="https://github.com/user-attachments/assets/a979dd7b-a03a-4fe2-8fe4-83fc7235878a" />
Anomaly image mask (for reference)
<img width="200" height="200" alt="005_mask" src="https://github.com/user-attachments/assets/35e1c8f0-2de5-4093-a8e8-8432f549d092" />


# Approach Overview
This project implements reconstruction-based anomaly detection using a U-Net–style convolutional autoencoder.

Core Idea:
Train the model only on normal images → it learns the distribution of "good" appearance → anomalies produce higher reconstruction error.

Pipeline:

- Train U-Net autoencoder on good samples

- Run inference on unseen images

- Compute per-pixel reconstruction error

- Generate anomaly heatmap

- Aggregate anomaly score per image

- Evaluate using AUROC


# Model Architecture
The model is a U-Net–style convolutional autoencoder, not a plain autoencoder.

Key characteristics:

- Fully convolutional encoder–decoder

- Skip connections between encoder and decoder (U-Net inductive bias)

- Better preservation of fine textures

- Improved localization of defects in reconstruction error maps


Architecture design goals:

- Stable training

- Interpretable behavior

- Fast inference

- Suitable for real industrial pipelines


Loss function:

- L1 reconstruction loss

- SSIM loss for improved perceptual quality


# Training Setup
Framework: PyTorch

Device support:

- CPU

- CUDA (NVIDIA GPUs)

- Apple Silicon MPS (Mac M1/M2/M3/M4 supported)

Image size: 512 × 512

Optimizer: Adam

Learning rate: 1e-4

Batch size: 24

Epochs: 75

Training uses only: train/good/*.png

No anomaly labels are used during training.


# Evaluation
Evaluation is performed using the official MVTec test split:

- Good samples

- Multiple defect types

Metric used:

**AUROC (Image-level)**

Measures how well anomaly scores separate good vs anomalous images.

Result on **tile** category:

**AUROC ≈ 0.75**

This is a strong baseline for a reconstruction-based approach and provides a foundation for more advanced methods (PatchCore, PaDiM, DRAEM, FastFlow, etc.).

# Qualitative Results
During inference, the pipeline produces:

- Original input image

- Reconstructed image

- Pixel-wise anomaly heatmap

Typical behavior:

- Good samples → low reconstruction error

- Defective regions → localized high-intensity error

This makes the approach interpretable, a key requirement for real-world inspection systems.


# How to run

**Install dependencies**

``` pip install -r requirements```

**train model**

```python train.py ```

**Run Inference**

```python infer.py ```

**Evaluate AUROC**

```python eval.py ```


# Possible Extensions

Natural next steps for improving performance and sophistication:

- Patch-based methods (PatchCore, PaDiM)

- Perceptual losses (VGG feature loss)

- Vision Transformer backbones

- Self-supervised pretraining

- Training across all MVTec categories

- Real-time inference demo (Streamlit / Gradio)

- ONNX export for deployment

# Author
Ayush Chauhan— Applied ML / Industrial Vision / NDE

This project intentionally prioritizes clarity, correctness, and realism over leaderboard chasing, reflecting real-world industrial ML practice.

# License
MIT License
