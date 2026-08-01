# Notice on Licensing Scope

The MIT License in this repository applies only to the source code contained herein (training scripts, evaluation scripts, augmentation pipeline, and other Python files), not to any dataset or model weights.

## Datasets

This project uses two third-party datasets, neither of which is distributed with this repository and neither of which is covered by the MIT License:

- **COD10K-v3**: subject to its original license terms from the dataset authors. See https://github.com/DengPingFan/SINet for access and terms.
- **Kvasir-SEG**: use is restricted to research and educational purposes. Commercial use is forbidden without prior written permission from Simula. See https://datasets.simula.no/kvasir-seg/ for the full terms.

## Pretrained Model Checkpoints

The pretrained model checkpoints distributed via the links in this README (COD10K baseline, Kvasir baseline, and both federated global models) were trained in part on the Kvasir-SEG dataset described above. As such, these checkpoints are provided for research and educational purposes only, consistent with the Kvasir-SEG dataset's usage terms, and are not covered by the MIT License. Commercial use of these checkpoints may require separate permission from Simula.
