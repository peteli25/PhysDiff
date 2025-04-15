# PhysDiff: A Physically-Guided Diffusion Model for Multivariate Time Series Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A physics-informed deep learning framework for detecting anomalies in multivariate time series data. This approach integrates wavelet transforms, Hilbert-Huang Transform (HHT), and diffusion models to capture complex temporal patterns and identify anomalies effectively.

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

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/dddlli/PhysDiff.git
cd PhysDiff

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

1. **Physics Encoder**: Extracts physics-guided features from time series using:
   - Wavelet transform for multi-scale analysis
   - Hilbert-Huang Transform for non-stationary pattern detection
   - Amplitude-Aware Permutation Entropy for complexity measurement

2. **Diffusion Model**: A denoising diffusion probabilistic model conditioned on physics features:
   - Forward process: Gradually adds noise to time series data
   - Reverse process: Learns to reconstruct normal patterns from noisy data

3. **Anomaly Detection**: Identifies anomalies using reconstruction error:
   - Normal data: Low reconstruction error
   - Anomalies: High reconstruction error
   - SPOT algorithm: Adaptive threshold setting

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

![Anomaly Visualization](./assets/anomaly_visualization.png)


## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This work builds upon advances in diffusion models for time series
- The SPOT algorithm implementation is adapted from the anomaly detection literature
- The Hilbert-Huang Transform implementation is based on the PyTorch HHT package

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{pgtsad2023,
  author = {Your Name},
  title = {PhysDiff: A Physically-Guided Diffusion Model for Multivariate Time Series Anomaly Detection},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dddlli/PhysDiff.git}}
}
```

## 📧 Contact

For questions or support, please open an issue on the GitHub repository or contact the authors.