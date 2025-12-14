"""
Topological Loss Landscape Analysis.

A framework for predicting neural network initialization quality
using topological features of the loss landscape.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

from .networks import SmallCNN, TinyCNN, MicroCNN, get_target_network
from .data_utils import load_cifar10, get_fixed_batch, compute_loss
from .init_data_generator import InitializationDataGenerator
from .loss_predictor import LossPredictor, train_loss_predictor
from .topology import (
    LossLandscapeSampler,
    TopologicalFeatureExtractor,
    extract_numerical_features,
    get_feature_names
)
from .quality_predictor import (
    QualityPredictor,
    train_quality_predictor,
    evaluate_initialization
)
from .baselines import run_baseline_comparisons
from .visualization import (
    plot_loss_predictor_training,
    plot_loss_predictor_evaluation,
    plot_persistence_diagrams,
    plot_loss_landscape_2d,
    plot_quality_prediction_results,
    plot_baseline_comparison,
    create_paper_figures
)

__version__ = "1.0.0"
__authors__ = "Apurva Mishra and Ayush Tibrewal"
