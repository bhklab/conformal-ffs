

#!/usr/bin/env python3
"""
Example script for running floating feature selection.
"""

import sys
import os
import json
import pickle
from datetime import datetime

# Add the package directory to Python path for direct import
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'package'))
sys.path.insert(0, package_dir)

# Import directly from the ffs module
import ffs

# Configuration
run_id = 0              # Fixed seed for reproducibility


#data_path = "synthetic"  # Use an internal synthetic dataset (for development purposes)
data_path = "data/melanoma_batch_corrected_data_common_genes_cosmic_io.csv"
n_experiments = 10      # Number of experiments to run

all_results = {}
for i in range(n_experiments):
    print(f"Running experiment {i+1}/{n_experiments} with run_id={run_id}")
    # Run the experiment    
    ffs_instance = ffs.FloatingFeatureSelector(run_id=run_id, data_path=data_path)
    experiment_result = ffs_instance.run_ffs(n_feat=12)  # Specify number of features to select before the optimization step

    print("Experiment completed successfully!")

    print("Selected features:", experiment_result)

    all_results[i+1] = {"selected_features": experiment_result, "run_id": i+1}

print("All experiments completed!")
print(all_results)

# Save results to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Save as JSON (human-readable)
json_file = os.path.join(results_dir, f"ffs_results_{timestamp}.json")
try:
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to JSON: {json_file}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# Save as pickle (preserves Python objects)
pickle_file = os.path.join(results_dir, f"ffs_results_{timestamp}.pkl")
try:
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to pickle: {pickle_file}")
except Exception as e:
    print(f"Error saving pickle: {e}")

print("Results successfully saved!")
