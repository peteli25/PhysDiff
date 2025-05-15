# PhysDiff: A Physically-Guided Diffusion Model for Multivariate Time Series Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Unsupervised anomaly detection for multivariate time series remains challenging due to complex nonstationary dynamics, high false positive rates, and limited interpretability. To address these issues, PhysDiff employs a two-stage process: physics-guided decomposition and diffusion-based reconstruction. Signal decomposition is necessary to disentangle overlapping dynamics by isolating high frequency oscillations and low frequency trends, which reduces interference and provides meaningful physical priors. Reconstruction through conditional diffusion modeling then captures deviations from learned normal behavior, making anomalies more distinguishable. First, we introduce an amplitude-sensitive permutation entropy criterion to adaptively determine the optimal decomposition depth, extracting frequency components without manual tuning. These components serve as explicit physical constraints. Second, we design a dual path conditional diffusion network that integrates decomposed signals and dynamically regulates denoising via a novel time frequency energy routing mechanism. By weighting reconstruction errors across frequency bands, our method improves anomaly localization and enhances interpretability. Extensive experiments on five benchmark datasets and two NeurIPS-TS scenarios demonstrate that PhysDiff outperforms 18 state-of-the-art baselines, with average F1-score improvements on both standard and challenging datasets. These results validate the necessity of combining principled signal decomposition with diffusion-based reconstruction for robust, interpretable anomaly detection in complex dynamic systems.

## ✨ Features

- **Physics-Guided Feature Extraction**: Extracts high and low-frequency components using Multi-channel Adaptive Fourier Decomposition (MAFD)
- **Complexity Measurement**: Amplitude-Sensitive Permutation Entropy (ASPE) for measuring time series complexity
- **Diffusion-Based Detection**: Uses physically-constrained diffusion models with Langevin dynamics for anomaly detection
- **Transformer Architecture**: Incorporates spatial-temporal transformer blocks with routing attention
- **Visualization Tools**: Comprehensive visualization for model understanding and anomaly interpretation

## 📋 Requirements

- Python 3.10+
- PyTorch 2.1+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

## 📊 Data Preparation

Organize your dataset in the following structure:

```
dataset/
└── YOUR_DATASET/
    ├── YOUR_DATASET_train_data.npy  # Training data (normal samples)
    ├── YOUR_DATASET_train_date.npy  # Training timestamps (optional)
    ├── YOUR_DATASET_test_data.npy   # Test data
    ├── YOUR_DATASET_test_date.npy   # Test timestamps (optional)
    └── YOUR_DATASET_test_label.npy  # Test labels (0: normal, 1: anomaly)
```

## 🏃‍♂️ Usage

### Basic Training and Testing

```bash
python main.py --dataset YOUR_DATASET
```

### Advanced Options

```bash
python main.py \
  --dataset YOUR_DATASET \
  --data_dir ./dataset/ \
  --model_dir ./checkpoint/ \
  --batch_size 64 \
  --epochs 20 \
  --window_size 64 \
  --model_dim 512 \
  --mafd_components 6 \
  --lambda_aspe 0.2 \
  --time_steps 1000 \
  --t 500 \
  --block_num 2 \
  --head_num 8 \
  --q 0.01 \
  --visualize
```

### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset name |
| `--data_dir` | Data directory path |
| `--model_dir` | Directory to save models and results |
| `--epochs` | Number of training epochs |
| `--batch_size` | Training batch size |
| `--lr` | Learning rate |
| `--window_size` | Size of sliding window |
| `--hidden_dim` | Hidden dimension size |
| `--physics_dim` | Physics feature dimension |
| `--mafd_components` | Number of MAFD components |
| `--lambda_aspe` | Weight for ASPE loss |
| `--time_steps` | Number of diffusion steps |
| `--q` | Risk level for SPOT algorithm |
| `--visualize` | Enable result visualization |

## 💡 Model Architecture

![Model Architecture](./assets/model_architecture.png)

The PhysDiff model consists of three main components:

1. **Physics-Guided Feature Extraction**:  We extract interpretable features via adaptive multi-scale signal decomposition, capturing both transient dynamics and long-term trends.

2. **Physically-Informed Diffusion Model**: We employ a conditional generative diffusion process incorporating these physical priors to robustly learn the distribution of normal patterns

3. **Anomaly Detection Scoring Module**: where anomalies are detected by contrasting reconstructed signals against observed data, with mechanisms sensitive to both point anomalies and sequence-level pattern deviations.

## 📂 Project Structure

```
PhysDiff/
├── data/
│   ├── dataset.py          # Dataset class for loading data
│   └── preprocess.py       # Data preprocessing functions
├── models/
│   ├── aspe.py             # Amplitude-Sensitive Permutation Entropy
│   ├── attention.py        # Attention mechanisms
│   ├── block.py            # Transformer blocks
│   ├── decomposition.py    # Dynamic decomposition for time series
│   ├── detector.py         # Main anomaly detection model
│   ├── diffusion.py        # Diffusion model implementation
│   ├── embedding.py        # Embedding layers
│   ├── mafd.py             # Multi-channel Adaptive Fourier Decomposition
│   ├── reconstruction.py   # Signal reconstruction module
│   └── subtraction.py      # Offset subtraction module
├── utils/
│   ├── earlystop.py        # Early stopping implementation
│   ├── evaluate.py         # Evaluation metrics and functions
│   ├── seed.py             # Random seed utilities
│   ├── spot.py             # SPOT algorithm for thresholding
│   └── visualization.py    # Visualization utilities
├── exp/
│   └── solver.py           # Experiment solver
├── main.py                 # Main script for running experiments
└── requirements.txt        # Project dependencies
```

## 📊 Evaluation and Visualization

After running the model, results will be displayed in the console:

```
=============== Results for YOUR_DATASET ===============
Precision: 0.9245
Recall: 0.8764
F1-Score: 0.8998
=======================================
```

If the `--visualize` flag is enabled, visualizations will be saved to the `checkpoint/` directory:


## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This work builds upon advances in diffusion models for time series
- The SPOT algorithm implementation is adapted from the anomaly detection literature

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{physdiff2025,
  author = {Anonymous Author(s)},
  title = {PhysDiff: A Physically-Guided Diffusion Model for Multivariate Time Series Anomaly Detection},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://anonymous.4open.science/r/PhysDiff-4726}}
}
```

## 📧 Contact

For questions or support, please open an issue on the GitHub repository or contact the authors.