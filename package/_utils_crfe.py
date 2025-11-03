"""
This module provides utility functions for the Conformal Recursive Feature Elimination
algorithm with standardized naming conventions and performance improvements.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Union, List, Optional
from numba import jit, njit

from scipy.stats import zscore
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings("ignore")

@njit
def _find_argmax_fast(beta_values: np.ndarray) -> int:
    """Optimized argmax finding using numba JIT compilation."""
    return np.argmax(beta_values)


def to_list(data: Union[List, np.ndarray]) -> List:
    """
    Convert array-like data to list format.
    
    Parameters
    ----------
    data : array-like
        Input data to convert
        
    Returns
    -------
    List
        Data in list format
        
    Raises
    ------
    ValueError
        If input is not a valid list or array
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):  
        return data.tolist()
    else:
        raise ValueError("Error: input must be a valid list or array")


def binary_change(y_train: np.ndarray, y_cal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary label transformation for two-class problems.
    
    Transforms binary labels to ensure consistent encoding with values {-1, 1}.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels
    y_cal : np.ndarray
        Calibration labels
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Transformed training labels, calibration labels, and unique class names
    """
    # Efficient vectorized operations: map 0 -> -1, keep other values
    y_train_transformed = np.where(y_train == 0, -1, y_train)
    y_cal_transformed = np.where(y_cal == 0, -1, y_cal)
    
    # Get unique classes and re-encode labels consistently
    unique_classes, y_train_encoded = np.unique(y_train_transformed, return_inverse=True)
    print(f"Binary classes: {unique_classes}")

    return y_train_encoded, y_cal_transformed, unique_classes


def find_argmax(beta_values: Union[List, np.ndarray]) -> int:
    """
    Find index of maximum value in beta array.
    
    Parameters
    ----------
    beta_values : array-like
        Beta values to find maximum from
        
    Returns
    -------
    int
        Index of maximum value
    """
    return _find_argmax_fast(np.array(beta_values))


def create_artificial_dataset(n_samples: int = 350, n_informative_features: int = 10, 
                            n_classes: int = 4, n_random_features: int = 25, 
                            random_seed: int = 12345, 
                            normalization: str = "zscore") -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Generate artificial dataset with optimized operations.
    
    Parameters
    ----------
    n_samples : int, default=350
        Number of samples to generate
    n_informative_features : int, default=10  
        Number of informative features
    n_classes : int, default=4
        Number of classes
    n_random_features : int, default=25
        Number of random noise features to add
    random_seed : int, default=12345
        Random seed for reproducibility
    normalization : str, default="zscore"
        Normalization method: "zscore" or "minmax"
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List]
        Features, labels, and class names
    """
    
    # Generate base classification dataset
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_informative_features, 
        n_redundant=0, 
        n_classes=n_classes,
        n_informative=n_informative_features, 
        n_clusters_per_class=1, 
        class_sep=1.5, 
        flip_y=0.05, 
        scale=None, 
        random_state=random_seed, 
        weights=[0.25] * n_classes, 
        shuffle=True
    )
    
    # Add random noise features efficiently
    rng = np.random.RandomState(random_seed)
    random_features = rng.randint(10, size=(n_samples, n_random_features))
    X = np.hstack([X, random_features])
    
    # Apply normalization
    if normalization == "zscore":
        X = zscore(X, axis=1)
    else:
        X = MinMaxScaler().fit_transform(X)

    # Create consistent class names and labels
    unique_classes = sorted(np.unique(y))
    
    # Vectorized label conversion
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = np.array([class_to_idx[cls] for cls in y])

    print(f"Classes: {unique_classes}")

    return X, y_encoded, unique_classes


class DataReader:
    """
    Optimized data reader with better performance and standardized interface.
    
    This class provides methods to load data from various sources including
    synthetic datasets and CSV files.
    """
    
    def __init__(self):
        """Initialize the DataReader."""
        self.X = None
        self.y = None
        self.n_classes = None
        self.n_samples = None
        self.n_features = None

    def load_data(self, data_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data with optimized I/O operations.
        
        Parameters
        ----------
        data_path : str
            Path to data or "synthetic" for artificial data generation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Features, labels, and class names
            
        Raises
        ------
        ValueError
            If path is invalid
        FileNotFoundError
            If data files are not found
        """
        
        if not isinstance(data_path, str):
            raise ValueError("Error: data path must be a valid string")

        if data_path == "synthetic":
            # Generate synthetic dataset
            X, y, class_names = create_artificial_dataset(
                n_samples=350, 
                n_informative_features=10, 
                n_classes=4, 
                n_random_features=25, 
                random_seed=12345, 
                normalization="zscore"
            )
            
            self.n_classes = len(class_names)
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
 
        else:

            # Load from file
            full_data_path =  data_path

            print(f"Loading data from: {full_data_path}")
             
            # Efficient CSV reading with appropriate dtypes
            try:
                if target_column == "recist":
                    df = pd.read_csv(full_data_path, header=0, index_col = 0)
                else:
                    df = pd.read_csv(full_data_path, header=0)
                
                df = df.dropna(subset=[target_column])

                print("We consider the first column as labels and the rest as features!!!!")
                
                y = df[target_column].values
                X = df.drop(columns=[target_column]).values
                
                #X = df.iloc[:, 2:].values
                #y = df.iloc[:, 1].values 

                # Create a mapping from unique response labels to integers
                unique_labels = pd.Series(y).dropna().unique()
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                print(f"Label mapping: {label_map}")
                # Map the response labels in y to integers using label_map
                y = pd.Series(y).map(label_map).values
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Data files not found in {full_data_path}: {e}")

            print(f"Data shape: X={X.shape}, y={y.shape}")
            
            # Get unique classes and encode consistently
            class_names, y_encoded = np.unique(y, return_inverse=True)
            y = y_encoded

            self.n_classes = len(class_names)
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]

        # Store data for potential reuse
        self.X = X
        self.y = y

        return X, y, class_names
        
    def get_n_classes(self) -> Optional[int]:
        """Get number of classes."""
        return self.n_classes

    def get_n_samples(self) -> Optional[int]:
        """Get number of samples."""
        return self.n_samples

    def get_n_features(self) -> Optional[int]:
        """Get number of features."""
        return self.n_features


@njit
def _compute_ncm_optimized(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                          bias: np.ndarray, lambda_param: float, lambda_p_param: float, 
                          is_multiclass: bool) -> np.ndarray:
    """
    Optimized Non-Conformity Measure computation using numba.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Labels
    weights : np.ndarray
        Model weights
    bias : np.ndarray
        Model bias terms
    lambda_param : float
        Lambda parameter for multiclass
    lambda_p_param : float
        Lambda prime parameter for multiclass
    is_multiclass : bool
        Whether this is a multiclass problem
        
    Returns
    -------
    np.ndarray
        Non-conformity measures
    """
    if not is_multiclass:
        # Binary case - vectorized computation
        scores = np.sum(weights * X, axis=1) + bias
        result = -(y * scores)
        return result.astype(np.float64)
    else:
        # Multiclass case
        n_samples = X.shape[0]
        ncm_values = np.zeros(n_samples, dtype=np.float64)
        
        for i in range(n_samples):
            y_label = int(y[i])
            
            # Compute first term: -lambda * (w_y^T x + b_y)
            term1 = -lambda_param * (np.dot(weights[y_label], X[i]) + bias[y_label])
            
            # Sum over all classes except y_label
            term2_sum = 0.0
            for k in range(weights.shape[0]):
                if k != y_label:
                    term2_sum += np.dot(weights[k], X[i]) + bias[k]
            
            # Second term: lambda_p * sum_{k != y}(w_k^T x + b_k)
            term2 = lambda_p_param * term2_sum
            ncm_values[i] = term1 + term2
            
        return ncm_values


def compute_nonconformity_measures(X: np.ndarray, y: np.ndarray, 
                                 weights: np.ndarray, bias: np.ndarray, 
                                 lambda_param: Optional[float] = None, 
                                 lambda_p_param: Optional[float] = None, 
                                 is_multiclass: bool = False) -> List[float]:
    """
    Compute Non-Conformity Measures with optimized implementation.
    
    This function provides backwards compatibility with the original API while using
    optimized implementations under the hood.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Labels
    weights : np.ndarray
        Model weights/coefficients
    bias : np.ndarray
        Model bias terms
    lambda_param : float, optional
        Lambda parameter for multiclass problems
    lambda_p_param : float, optional  
        Lambda prime parameter for multiclass problems
    is_multiclass : bool, default=False
        Whether this is a multiclass problem
        
    Returns
    -------
    List[float]
        Non-conformity measure values
        
    Raises
    ------
    ValueError
        If lambda parameters are missing for multiclass problems
    """
    # Ensure consistent data types
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    
    # Handle different input formats for y
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    
    if not is_multiclass:
        # Binary classification case
        y = y.astype(np.float64)
        scores = np.sum(weights * X, axis=1) + bias
        return (-(y * scores)).tolist()
    else:
        # Multiclass case - validate parameters
        y = y.astype(np.int32)
        if lambda_param is None or lambda_p_param is None:
            raise ValueError("Lambda parameters must be provided for multiclass problems")
        
        # Use the optimized numba function
        result = _compute_ncm_optimized(X, y, weights, bias, lambda_param, lambda_p_param, True)
        return result.tolist()


# Backwards compatibility aliases
NC_OvsA_SVMl_dev = compute_nonconformity_measures
READER = DataReader  # Backwards compatibility alias
