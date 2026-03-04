"""
Feature engineering utilities for Wasserstein risk analysis.
"""

import numpy as np
import pandas as pd


def normalize_features(X):
    """
    Normalize features to zero mean and unit variance.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features.
    
    Returns
    -------
    np.ndarray
        Normalized features.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


def scale_features(X, min_val=0, max_val=1):
    """
    Scale features to a specified range.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features.
    min_val : float, default=0
        Minimum value of the scaled range.
    max_val : float, default=1
        Maximum value of the scaled range.
    
    Returns
    -------
    np.ndarray
        Scaled features.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
    return X_scaled * (max_val - min_val) + min_val
