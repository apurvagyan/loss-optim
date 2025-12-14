# Topological Loss Landscape Analysis for Neural Network Initialization

**CPSC 6440 Final Project**  
**Authors:** Apurva Mishra and Ayush Tibrewal  
**Course:** Geometric and Topological Methods in Machine Learning

## Overview

This project develops a topological framework for predicting the quality of neural network initializations. We use persistent homology to characterize the local structure of loss landscapes and predict training outcomes.

### Key Contributions

1. **Loss Predictor Network**: A neural network that learns to approximate the loss landscape, enabling efficient sampling without expensive forward passes.

2. **Topological Feature Extraction**: We compute persistent homology on sampled loss landscapes, extracting features like Betti curves, persistence entropy, and lifetime statistics.

3. **Quality Prediction**: Using topological features, we predict how well an initialization will perform after training.

## Project Structure

```
topo_loss_landscape/
├── configs/
│   └── config.yaml           # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── networks.py           # Target network architectures
│   ├── data_utils.py         # Data loading utilities
│   ├── init_data_generator.py # Generate (params, loss) pairs
│   ├── loss_predictor.py     # Stage 1: Loss predictor network
│   ├── topology.py           # Stage 2: Topological feature extraction
│   ├── quality_predictor.py  # Stage 2: Quality prediction
│   ├── baselines.py          # Baseline methods for comparison
│   └── visualization.py      # Plotting utilities
├── scripts/
│   └── run_pipeline.py       # Main execution script
├── data/                     # Data directory (created automatically)
├── checkpoints/              # Model checkpoints
├── figures/                  # Generated figures
├── outputs/                  # Results and logs
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd topo_loss_landscape

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

Key dependencies:
- `torch`, `torchvision`: Deep learning
- `ripser`, `persim`, `giotto-tda`, `gudhi`: Persistent homology
- `scikit-learn`: Machine learning utilities
- `matplotlib`, `seaborn`: Visualization

## Usage

### Quick Start

Run the complete pipeline:

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

### Stage-by-Stage Execution

```bash
# Stage 1 only: Generate data and train loss predictor
python scripts/run_pipeline.py --config configs/config.yaml --stage 1

# Stage 2 only: Topological analysis (requires Stage 1 checkpoint)
python scripts/run_pipeline.py --config configs/config.yaml --stage 2
```

### Configuration

Edit `configs/config.yaml` to adjust:

- **Data generation**: Number of samples, perturbation scale
- **Loss predictor**: Architecture, training hyperparameters
- **Topology**: Sampling parameters, homology dimension
- **Quality prediction**: Number of evaluations, training epochs

## Method

### Stage 1: Loss Predictor

We train a neural network $f_\theta: \mathbb{R}^d \to \mathbb{R}$ that maps flattened network parameters to their corresponding loss values:

$$f_\theta(\mathbf{w}) \approx \mathcal{L}(\mathbf{w})$$

This is trained on 100K+ samples of (parameters, loss) pairs generated from various initialization schemes.

### Stage 2: Topological Analysis

For each initialization $\mathbf{w}_0$, we:

1. **Sample the local landscape**: Generate points $\{\mathbf{w}_0 + \epsilon_i\}$ and predict their losses
2. **Compute persistent homology**: 
   - Vietoris-Rips filtration on the point cloud
   - Sublevel set filtration using loss as height function
3. **Extract features**: Betti curves, persistence lifetimes, entropy

### Quality Prediction

A smaller network predicts test loss after training from topological features:

$$\text{Quality}(\mathbf{w}_0) = g_\phi(\text{TopoFeatures}(\mathbf{w}_0))$$

## Results

Our method achieves significant correlation between predicted and actual initialization quality, outperforming baselines that use only:
- Initial loss value
- Local loss statistics (mean, std)
- Linear models on features

See `figures/` for generated plots after running the pipeline.

## Reproducing Results

1. Results are saved to:
   - `checkpoints/`: Model weights
   - `figures/`: All plots
   - `data/`: Generated datasets

2. For faster iteration, reduce `n_samples` in config.

Command structure to follow includes:

```bash
   # full run
   python scripts/01_generate_init_data.py --n_samples 100000
   python scripts/02_train_loss_predictor.py --epochs 200
   python scripts/03_generate_quality_data.py --n_samples 200
   python scripts/04_train_quality_predictor.py --epochs 150
   python scripts/05_generate_figures.py --format pdf

   # quick test
   python scripts/01_generate_init_data.py --n_samples 5000
   python scripts/02_train_loss_predictor.py --epochs 50
   python scripts/03_generate_quality_data.py --n_samples 50
   python scripts/04_train_quality_predictor.py --epochs 50
   ```

## Citation

If you use this code, please cite:

```bibtex
@misc{mishra2025topoloss,
  title={Learning to Predict and Optimize Neural Network Training Through Topological Loss Landscape Analysis},
  author={Mishra, Apurva and Tibrewal, Ayush},
  year={2025},
  note={CPSC 6440 Final Project, Yale University}
}
```

## References

1. Carlsson, G. (2009). Topology and data. *Bulletin of the AMS*.
2. Rieck, B., et al. (2019). Neural persistence: A complexity measure for deep neural networks using algebraic topology. *ICLR*.
3. Li, H., et al. (2018). Visualizing the loss landscape of neural nets. *NeurIPS*.

## License

MIT License - See LICENSE file for details.
