"""
Core analysis utilities for Wasserstein risk computation.
"""

import numpy as np
from scipy.spatial.distance import cdist


def wasserstein_distance(X, Y, p=2):
    """
    Compute the Wasserstein distance between two distributions.
    
    Parameters
    ----------
    X : np.ndarray
        Samples from the first distribution, shape (n_samples, n_features).
    Y : np.ndarray
        Samples from the second distribution, shape (m_samples, n_features).
    p : int, default=2
        Order of the Wasserstein distance (e.g., 2 for 2-Wasserstein).
    
    Returns
    -------
    float
        The Wasserstein distance between X and Y.
    """
    # Compute pairwise distances
    distances = cdist(X, Y, metric='euclidean')
    
    # For simplicity, return the mean minimum distance as an approximation
    # In practice, you'd use optimal transport solvers like POT
    W_dist = np.mean(np.min(distances, axis=1))
    
    return W_dist ** (1/p) if p > 1 else W_dist


def compute_risk_metrics(data, ground_truth=None):
    """
    Compute risk metrics from data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data.
    ground_truth : np.ndarray, optional
        Ground truth labels or values.
    
    Returns
    -------
    dict
        Dictionary containing computed risk metrics.
    """
    metrics = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
    }
    
    if ground_truth is not None:
        metrics['mae'] = np.mean(np.abs(data - ground_truth))
    
    return metrics
