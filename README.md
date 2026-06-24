# Cross-Domain Federated Segmentation

A federated learning framework using FedProx and Pyramid Vision Transformer (PVTv2) backbones to train robust joint segmentation models across medical imaging and camouflaged object detection domains without performance degradation or data sharing.

## Problem Statement

Deep learning models for semantic segmentation suffer from catastrophic generalization failure when deployed on out-of-distribution domains. For instance, a model trained solely on medical polyps fails on camouflaged objects, and vice versa. Standard Federated Learning (FedAvg) also struggles to converge when clients possess highly heterogeneous features and labels. This project uses federated learning to build a unified segmentation model that generalizes across medical and camouflage domains while preserving data privacy.

## Architecture Overview

- **Model Backbone**: `CamouflageSegNet` utilizing a Pyramid Vision Transformer (`pvt_v2_b2`) encoder pretrained on ImageNet.
- **Decoder Architecture**: A lightweight custom decoder with convolutional bottleneck layers, lateral feature interpolation/concatenation, and dual prediction heads for segmentation and boundary edge map extraction.
- **Federated Algorithm**: `FedProx` framework featuring client-side regularization with a proximal term ($\mu = 0.01$) to stabilize aggregation.
- **Training Flow**: 
  1. The server broadcasts the global `CamouflageSegNet` weights to two client nodes.
  2. **Client 1 (COD10K)**: Trains locally on camouflage images using segment and edge loss.
  3. **Client 2 (Kvasir-SEG)**: Trains locally on polyp images using segment and edge loss.
  4. The server aggregates model parameters via weighted average (FedAvg weight parameters: 0.75 for Client 1 and 0.25 for Client 2).

## Datasets Used

- **COD10K-v3**: A large-scale Camouflaged Object Detection dataset comprising 3,040 training and test images annotated with semantic masks and edge maps.
- **Kvasir-SEG**: A medical polyp segmentation dataset containing 1,000 gastrointestinal images and corresponding ground-truth masks.

### Citations

If you use these datasets, please cite their original papers:

**COD10K**:
```bibtex
@inproceedings{fan2020camouflaged,
  title={Camouflaged object detection},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2777--2787},
  year={2020}
}
```

**Kvasir-SEG**:
```bibtex
@inproceedings{jha2020kvasir,
  title={Kvasir-seg: A segmented polyp dataset and baseline for medical image segmentation},
  author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
  booktitle={International Conference on MultiMedia Modeling (MMM)},
  pages={451--462},
  year={2020},
  organization={Springer}
}
```

## Dataset Setup

Download the datasets and place their zip archives directly into the project root directory:

1. **COD10K-v3**: Download `COD10K-v3.zip` and place it in the root folder. The training scripts automatically extract this to `./COD10K-v3/` if the folder does not exist.
2. **Kvasir-SEG**: Download `kvasir-seg.zip` and place it in the root folder. The training scripts automatically extract this to `./Kvasir-SEG/` if the folder does not exist.

Ensure your root directory structure is laid out as follows:
```text
.
├── COD10K-v3/
│   ├── Train/
│   │   ├── Image/
│   │   └── GT_Object/
│   └── Test/
│       ├── Image/
│       └── GT_Object/
├── Kvasir-SEG/
│   ├── images/
│   └── masks/
├── COD10K-v3.zip (optional)
├── kvasir-seg.zip (optional)
├── compare.py
├── fed_model_train.py
└── ...
```

## Results Table

Evaluation results on the respective test splits show that single-domain baselines fail catastrophically cross-domain, whereas the FL Global model maintains high, generalized performance across both:

| Model | COD10K Test (Dice) | Kvasir Test (Dice) |
| :--- | :---: | :---: |
| **COD10K-only Baseline** | 0.8248 | 0.1506 |
| **Kvasir-only Baseline** | 0.2239 | 0.8950 |
| **FL Global (FedProx)** | **0.8132** | **0.8893** |

## How to Run

### 1. Installation

Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Standalone Training

To train the single-domain baselines locally:
- **Kvasir Baseline**:
  ```bash
  python kvasir_baseline_train.py
  ```
- **COD10K Baseline**:
  ```bash
  python cod10k_baseline_train.py
  ```

### 3. Federated Training

To start or resume the Federated Learning process (which outputs checkpoints to `fl_checkpoints/`):
```bash
python fed_model_train.py
```

### 4. Cross-Domain Evaluation

To run the complete cross-domain evaluation matrix and output `cross_domain_results.json`:
```bash
python compare.py
```

## Pretrained Weights Note

Model checkpoints available on request — contact via [23211a66j0@gmail.com] or raise an issue.

## Project Status

**Ongoing** (actively improving Kvasir-SEG results via augmentation).
