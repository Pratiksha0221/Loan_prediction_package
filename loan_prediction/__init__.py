"""
Loan Prediction Package

This package includes modules for loading, preprocessing, training models, and visualizing results.
"""

from .data_loader import load_data
from .data_preprocessing import DataProcessor
from .model import ModelTrainer
from .visualization import Visualizer

# Define what is accessible when importing the package
__all__ = [
    "load_data",
    "DataProcessor",
    "ModelTrainer",
    "Visualizer"
]
