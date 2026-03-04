"""
Data loading and preprocessing utilities for Wasserstein risk analysis.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    """
    Preprocess data for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """
    # Remove any rows with missing values
    df = df.dropna()
    return df
