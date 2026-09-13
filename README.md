# Kaggle Training Pipelines

Unified repository containing automated Kaggle training notebooks for the
LemGendary machine learning model suite.

## Overview

This repository hosts 22 standalone Jupyter notebooks engineered for high-speed
cloud training on Kaggle GPU/TPU accelerators. Each notebook incorporates:

- **Hardware Sentinel**: Verifies accelerator architecture and PyTorch CUDA
  compatibility.
- **Automated Dependency Synchronization**: Installs and pins environment
  packages required for the specific model architecture.
- **Dataset Ingestion**: Connects to Kaggle Datasets and cloud storage
  repositories for seamless manifold streaming.
- **Checkpoint & Artifact Export**: Implements automated export to Hugging Face
  Hub, Kaggle Models, and local artifacts.

## Notebook Inventory

| Notebook | Architecture / Task |
| --- | --- |
| `codeformer_training.ipynb` | Blind Face Restoration |
| `ffanet_indoor_training.ipynb` | Single Image Dehazing (Indoor) |
| `ffanet_outdoor_training.ipynb` | Single Image Dehazing (Outdoor) |
| `film_restorer_training.ipynb` | Frame Interpolation (FILM) |
| `forex_predictor_training.ipynb` | Multi-Timeframe Forex Forecasting |
| `mirnet_exposure_training.ipynb` | Dual-Residual Exposure Correction |
| `mirnet_lowlight_training.ipynb` | Low-Light Enhancement |
| `mprnet_deraining_training.ipynb` | Progressive Restoration (Deraining) |
| `nafnet_debluring_training.ipynb` | NAFNet Image Deblurring |
| `nafnet_denoising_training.ipynb` | NAFNet Image Denoising |
| `nima_aesthetic_efficientnet_training.ipynb` | Aesthetics (EfficientNet) |
| `nima_aesthetic_mobile_training.ipynb` | NIMA Aesthetics (MobileNet) |
| `nima_aesthetic_pro_training.ipynb` | NIMA Aesthetics (Pro Grade) |
| `nima_authenticity_training.ipynb` | NIMA Authenticity Detection |
| `nima_technical_training.ipynb` | NIMA Technical Assessment |
| `parsenet_training.ipynb` | Face Parsing & Segmentation |
| `professional_multitask_restoration_training.ipynb` | Restoration Engine |
| `retinaface_training.ipynb` | Face Landmark Localization |
| `ultrazoom_training.ipynb` | Super-Resolution & Ultrazoom |
| `universal_nsfw_classification_training.ipynb` | NSFW Filter Classification |
| `upn_v2_training.ipynb` | Perceptual Super-Resolution |
| `yolov8n_training.ipynb` | Object Detection & Bounding |

## Execution Guidelines

1. Push or import notebooks to your Kaggle account.
2. Under **Notebook Settings**, select accelerator:
   - **GPU P100** or **GPU T4 x2** (recommended).
3. Ensure **Internet access** is turned **On** in settings.
4. Run all cells sequentially.
