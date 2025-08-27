
"""
Floating Feature Selection (FFS) - Main execution module.

This module provides the main entry point for running feature selection experiments
using CRFE (Conformal Recursive Feature Elimination) and mRMR methods with
standardized naming conventions and improved performance.

Authors: Marcos López De Castro 
         Alberto García Galindo
         Rubén Armañanzas
"""

import numpy as np
import sys
import os
import json
import argparse
import pickle
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

from _utils_crfe import DataReader
from _crfe import CRFE
from _MrmrMS import FeatureSelector
from _utils import predict_scores_svm


# Cache random number generator for efficiency
@lru_cache(maxsize=128)
def _get_random_state(seed: int) -> np.random.Generator:
    """Cached random state generator to avoid recreation."""
    return np.random.default_rng(seed)


def generate_random_integer(seed: int) -> int:
    """
    Generate random integer with caching for efficiency.
    
    Parameters
    ----------
    seed : int
        Random seed
        
    Returns
    -------
    int
        Random integer between 0 and 99999
    """
    rng = _get_random_state(seed)
    return rng.integers(low=0, high=100000)


def create_linear_svc_estimator() -> LinearSVC:
    """
    Factory function for creating optimized LinearSVC estimator.
    
    Returns
    -------
    LinearSVC
        Configured LinearSVC estimator
    """
    return LinearSVC(
        tol=1e-4, 
        loss='squared_hinge',
        max_iter=14000,
        dual="auto"
    )



def split_dataset(X: np.ndarray, y: np.ndarray, run_id: int, 
                 test_size: float = 0.15, cal_size: float = 0.5) -> Tuple[np.ndarray, ...]:
    """
    Split data into training, calibration, and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target labels
    run_id : int
        Run identifier for reproducible splits
    test_size : float, default=0.15
        Proportion of data to use for testing
    cal_size : float, default=0.5
        Proportion of remaining data to use for calibration
        
    Returns
    -------
    Tuple[np.ndarray, ...]
        X_train, X_cal, X_test, y_train, y_cal, y_test
        
    Raises
    ------
    ValueError
        If data shapes are inconsistent or sizes are invalid
    """
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if not (0 < cal_size < 1):
        raise ValueError("cal_size must be between 0 and 1")

    # Generate deterministic seed based on run_id
    seed = generate_random_integer(42 + run_id)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=True, 
        stratify=y, 
        random_state=seed
    )
    
    # Second split: separate training and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, 
        test_size=cal_size, 
        shuffle=True, 
        stratify=y_temp, 
        random_state=seed
    )
    
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def run_crfe_experiment(estimator, X_train: np.ndarray, y_train: np.ndarray, 
                       X_cal: np.ndarray, y_cal: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    
    """
    Run CRFE experiment with the given data splits.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator for feature selection
    X_train, y_train : Training data and labels
    X_cal, y_cal : Calibration data and labels
    X_test, y_test : Test data and labels
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary or error information
    """
    
    n_features = X_train.shape[1]
    feature_indices = np.arange(n_features)
    
    if n_features <= 1:
        raise ValueError("Need at least 2 features for feature selection")
    
    # Create CRFE with standardized parameters
    crfe = CRFE(
        estimator=estimator, 
        n_features_to_select=n_features - 1, 
        lambda_param=0.5,  # Default value
        epsilon=0.4        # Default value
    )
    
    # Fit the CRFE model
    crfe.fit(X_train, y_train, X_cal, y_cal, X_test, y_test)

    list_of_selected_features = crfe.results_dict_

    removed_features = np.setdiff1d(feature_indices, list_of_selected_features)

    return removed_features



def run_mrmr_experiment(estimator, X_train: np.ndarray, y_train: np.ndarray, 
                       X_cal: np.ndarray, y_cal: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       selected_features) -> Dict[str, Any]:
    """
    Run mRMR-MS experiment with the given data splits.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator (unused in mRMR but kept for consistency)
    X_train, y_train : Training data and labels
    X_cal, y_cal : Calibration data and labels (unused)
    X_test, y_test : Test data and labels (unused)
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary or error information
    """
    selected_features = selected_features.tolist()

    try:
        n_classes = len(np.unique(y_train))
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Create mRMR feature selector
        mrmr = FeatureSelector(
            classes_=[i for i in range(n_classes)],
            max_features=len(selected_features)+1, 
            parallel=True,
            verbose=True
        )
        
        # mRMR-MS parameters
        kernel = "linear"     # Options: "linear", "rbf", "poly"
        split_size = 0.5      # Split size for mRMR-MS
        
        # Run mRMR-MS feature selection
        mrmr.mRMR_MS(X_train, y_train, kernel, split_size,  selected_features)

        return mrmr.all_selected_features[-1] if hasattr(mrmr, 'all_selected_features') else {}
    
    except Exception as e:
        print(f"Error in mRMR experiment: {str(e)}")
        return {'error': str(e)} 





class FloatingFeatureSelector:
    """
    Floating Feature Selection (FFS) experiment manager.
    
    This class provides a structured approach to running feature selection experiments
    using both CRFE (Conformal Recursive Feature Elimination) and mRMR methods.
    
    Attributes
    ----------
    run_id : int
        Run identifier for reproducibility
    data_path : str
        Path to data or "synthetic" for artificial data
    test_size : float
        Proportion of data to use for testing
    cal_size : float
        Proportion of remaining data to use for calibration
    estimator : sklearn estimator
        Base estimator for feature selection
    data_reader : DataReader
        Data loading utility
    experiment_results : Dict[str, Any]
        Storage for experiment results
    """
    
    def __init__(self, run_id: int = 1, data_path: str = "synthetic", 
                 test_size: float = 0.15, cal_size: float = 0.5,
                 estimator=None):
        """
        Initialize the Floating Feature Selector.
        
        Parameters
        ----------
        run_id : int, default=1
            Run identifier for reproducibility
        data_path : str, default="synthetic"
            Path to data or "synthetic" for artificial data
        test_size : float, default=0.15
            Proportion of data to use for testing
        cal_size : float, default=0.5
            Proportion of remaining data to use for calibration
        estimator : sklearn estimator, optional
            Base estimator for feature selection. If None, creates LinearSVC
        """
        self.run_id = run_id
        self.data_path = data_path
        self.test_size = test_size
        self.cal_size = cal_size
        self.estimator = estimator or create_linear_svc_estimator()
        self.data_reader = DataReader()
        
        # Initialize storage for experiment components
        self.X = None
        self.y = None
        self.class_names = None
        self.classes_ = None  # For sklearn compatibility
        self.lambda_param = 0.5  # Default lambda parameter
        self.lambda_p_param = None  # Will be calculated based on number of classes
        self.X_train = None
        self.X_cal = None
        self.X_test = None
        self.y_train = None
        self.y_cal = None
        self.y_test = None
        
        # Floating Feature Selection specific attributes
        self.S = None  # Selected features set
        self.U = None  # Available features set
        self.best_metric = None
        self.best_subset = None
        self.new_scores = None
        self.new_subsets = None  # Store corresponding subsets for new_scores
        self.counter = 0
        
        # Conformal prediction metrics
        self.Empirical_coverage_ = None
        self.Uncertainty_ = None
        self.Certainty_ = None
        
        self.experiment_results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data using the configured data reader.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Features, labels, and class names
        """
        self.X, self.y, self.class_names = self.data_reader.load_data(self.data_path)
        
        print(f"Dataset loaded: {self.X.shape} samples, {len(self.class_names)} classes: {self.class_names}")
        sys.stdout.flush()
        
        return self.X, self.y, self.class_names
    
    def split_data(self) -> Tuple[np.ndarray, ...]:
        """
        Split loaded data into training, calibration, and test sets.
        
        Returns
        -------
        Tuple[np.ndarray, ...]
            X_train, X_cal, X_test, y_train, y_cal, y_test
            
        Raises
        ------
        ValueError
            If data has not been loaded yet
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before splitting. Call load_data() first.")
            
        splits = split_dataset(self.X, self.y, self.run_id, self.test_size, self.cal_size)
        self.X_train, self.X_cal, self.X_test, self.y_train, self.y_cal, self.y_test = splits
        
        # Set classes and calculate lambda_p_param
        self.classes_ = np.unique(self.y_train)
        if len(self.classes_) > 2:
            self.lambda_p_param = (1 - self.lambda_param) / (len(self.classes_) - 1)
        else:
            self.lambda_p_param = 0.0
        
        print(f"Data split - Train: {self.X_train.shape}, Cal: {self.X_cal.shape}, Test: {self.X_test.shape}")
        sys.stdout.flush()
        
        return splits
    
    def init_S_U(self):
        """Initialize the selected (S) and unselected (U) feature sets."""
        
        if self.S is None:
            
            S = set(np.random.choice(
                self.X_train.shape[1], 
                size=int(self.X_train.shape[1] * 0.15), 
                replace=False
            ))

            all_features = set(range(self.X_train.shape[1]))
            U = all_features - S  # Complementary set

            self.S = np.array(sorted(S), dtype=int)
            self.U = np.array(sorted(U), dtype=int)


        print(f"Initial S (selected): {self.S} ")
        print(f"Initial U (unselected): {self.U} ")


        return None
    


    def update(self):
        """Update best metric and subset if current performance is better."""
        if self.best_metric is None:
            self.best_metric = self.Uncertainty_
            self.best_subset = self.S.copy()
            return None

        # Check if any of the new scores is better than current best
        if hasattr(self, 'new_scores') and self.new_scores is not None:
            counter = 0
            for ele in self.new_scores:
                if ele < self.best_metric:
                    self.best_metric = ele
                    if hasattr(self, 'new_subsets') and self.new_subsets is not None:
                        self.best_subset = self.new_subsets[counter].copy()
                    return counter             
                
                else:
                    counter += 1

        return None
    



    def init_method(self):
        """Initialize the floating feature selection method."""
        
        self.init_S_U()  # random init if not initialized

        # Calculate initial scores using the utility function

        X_train_init = self.X_train[:,  self.S]
        X_cal_init = self.X_cal[:,  self.S]  
        X_test_init = self.X_test[:, self.S]

        scores = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_init, self.y_train, X_cal_init, self.y_cal, X_test_init, self.y_test
        )

        self.Empirical_coverage_ = scores[0]
        self.Uncertainty_ = scores[1]
        self.Certainty_ = scores[2]

        self.update()

        return None
    
    
    def run_crfe_experiment(self) -> Dict[str, Any]:
        """
        Run CRFE (Conformal Recursive Feature Elimination) experiment.
        
        Returns
        -------
        Dict[str, Any]
            CRFE experiment results
            
        Raises
        ------
        ValueError
            If data has not been split yet
        """
        if self.X_train is None:
            raise ValueError("Data must be split before running experiments. Call split_data() first.")
            
        print("Running CRFE experiment...")
        sys.stdout.flush()


        X_train_S = self.X_train[:, self.S].copy()
        X_cal_S = self.X_cal[:, self.S].copy()
        X_test_S = self.X_test[:, self.S].copy()


        removed_feature = run_crfe_experiment(
            self.estimator, X_train_S, self.y_train, 
            X_cal_S, self.y_cal, X_test_S, self.y_test
        )
        

        f_removed_feature = self.S[removed_feature]
        print("Removed feature: ", f_removed_feature)
        
        return f_removed_feature
    
    def run_mrmr_experiment(self) -> Dict[str, Any]:
        """
        Run mRMR-MS (minimum Redundancy Maximum Relevance - Multi-class SVM) experiment.
        
        Returns
        -------
        Dict[str, Any]
            mRMR experiment results
            
        Raises
        ------
        ValueError
            If data has not been split yet
        """
        if self.X_train is None:
            raise ValueError("Data must be split before running experiments. Call split_data() first.")
            
        print("Running mRMR experiment...")
        sys.stdout.flush()

        X_train_U = self.X_train.copy()
        X_cal_U = self.X_cal.copy()
        X_test_U = self.X_test.copy()
        
        S = run_mrmr_experiment(
            self.estimator, X_train_U, self.y_train, 
            X_cal_U, self.y_cal, X_test_U, self.y_test, self.S
        )
        
        f_added_feature = [S[-1]]
        print(f"Added feature: {f_added_feature}")


        
        return f_added_feature
    
    def _evaluate_moves(self, f_removed, f_added):
        """
        Evaluate three potential moves: removal, addition, and swap.
        
        Parameters
        ----------
        f_removed : int
            Feature to be removed from current subset S
        f_added : int  
            Feature to be added to current subset S
            
        Returns
        -------
        dict
            Dictionary containing metrics for each move type
        """
        current_S = self.S.copy()
        
        # 1. Removal move: S_minus = S \ {f_removed}       
        S_minus = np.setdiff1d(current_S, f_removed)

        
        # Extract training, calibration, and test data for removal move
        X_train_minus = self.X_train[:, S_minus]
        X_cal_minus = self.X_cal[:, S_minus]  
        X_test_minus = self.X_test[:, S_minus]
        
        # Compute scores for removal move
        scores_minus = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_minus, self.y_train, X_cal_minus, self.y_cal, X_test_minus, self.y_test
        )
        metric_minus = scores_minus[1]  # Using Uncertainty as the metric
        
        # 2. Addition move: S_plus = S ∪ {f_added}  
        S_plus = np.append(current_S, f_added)
        
        # Extract training, calibration, and test data for addition move
        X_train_plus = self.X_train[:, S_plus]
        X_cal_plus = self.X_cal[:, S_plus]
        X_test_plus = self.X_test[:, S_plus]
        
        # Compute scores for addition move
        scores_plus = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_plus, self.y_train, X_cal_plus, self.y_cal, X_test_plus, self.y_test
        )
        metric_plus = scores_plus[1]  # Using Uncertainty as the metric
        
        # 3. Swap move: S_swap = (S \ {f_removed}) ∪ {f_added}
        S_swap = np.setdiff1d(current_S, f_removed)
        S_swap = np.append(S_swap, f_added)
        
        # Extract training, calibration, and test data for swap move  
        X_train_swap = self.X_train[:, S_swap]
        X_cal_swap = self.X_cal[:, S_swap]
        X_test_swap = self.X_test[:, S_swap]
        
        # Compute scores for swap move
        scores_swap = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_swap, self.y_train, X_cal_swap, self.y_cal, X_test_swap, self.y_test
        )
        metric_swap = scores_swap[1]  # Using Uncertainty as the metric
        
        # Store results
        move_results = {
            'removal': {
                'subset': S_minus,
                'metric': metric_minus,
                'scores': scores_minus
            },
            'addition': {
                'subset': S_plus, 
                'metric': metric_plus,
                'scores': scores_plus
            },
            'swap': {
                'subset': S_swap,
                'metric': metric_swap, 
                'scores': scores_swap
            }
        }
        
        print(f"Move evaluation results:")
        print(f"  Current best metric: {self.best_metric:.4f}")
        print(f"  Removal move metric: {metric_minus:.4f}")
        print(f"  Addition move metric: {metric_plus:.4f}")  
        print(f"  Swap move metric: {metric_swap:.4f}")

        # Store the new scores and corresponding subsets for update method
        self.new_scores = [metric_minus, metric_plus, metric_swap]
        self.new_subsets = [S_minus, S_plus, S_swap]
        
        # Try to update with new scores
        improvement_index = self.update()
        
        if improvement_index is not None:
            # An improvement was found
            move_names = ['removal', 'addition', 'swap']
            best_move_name = move_names[improvement_index]
            best_move_subset = self.new_subsets[improvement_index]
            best_move_scores = [scores_minus, scores_plus, scores_swap][improvement_index]
            
            print(f"  ✓ Improvement found! Best move: {best_move_name}")
            print(f"  ✓ New best metric: {self.best_metric:.4f}")
            
            # Update current feature sets
            self.S = best_move_subset.copy()
            all_features = set(range(self.X_train.shape[1]))
            self.U = np.array(sorted(all_features - set(self.S)), dtype=int)
            
            # Update current conformal prediction metrics
            self.Empirical_coverage_ = best_move_scores[0]
            self.Uncertainty_ = best_move_scores[1] 
            self.Certainty_ = best_move_scores[2]
            
            
            
            # Add improvement info to results
            #move_results['improvement'] = True
            #move_results['best_move'] = best_move_name
            #move_results['improvement_index'] = improvement_index
            
            return move_results
        

        else:
            print(f"  ✗ No improvement found. Keeping current subset.\n")
            move_results['improvement'] = False
            move_results['best_move'] = None
            return None  # Signal to stop the FFS loop
    
        
        
    
    def _run_ffs(self):

        # Run experiments
        for i in range(10):  # Limit to 10 iterations for safety
            f_removed = self.run_crfe_experiment()  #self.f_removed
            f_added = self.run_mrmr_experiment()  #self.f_added

            ret = self._evaluate_moves(f_removed, f_added)
            print("Current features selected: ", self.S, "\n")
            if ret is None:
                break
            
        return None
        
    
    def run_ffs(self) -> Dict[str, Any]:
        """
        Run feature selection experiments in sequence.
        
        This method handles the complete pipeline:
        1. Load data
        2. Split data
        3. Run CRFE experiment
        4. Run mRMR experiment
        
        Returns
        -------
        Dict[str, Any]
            Combined results from all experiments
        """
        # Load and prepare data
        self.load_data()
        self.split_data()
    

        # Initialize the floating feature selection method
        self.init_method()

        # Run your floating feature selection algorithm
        ffs_results = self._run_ffs()
        
        
        return None
    



if __name__ == "__main__":
    """Main execution block with standardized parameters."""
  
    run_id = 1              # Fixed seed for reproducibility
    data_path = "synthetic"  # Use synthetic dataset


        
    ffs = FloatingFeatureSelector(run_id=run_id, data_path=data_path)
    results = ffs.run_ffs()


    
    print("Experiment completed successfully!")
        

