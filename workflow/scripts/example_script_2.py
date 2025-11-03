

#!/usr/bin/env python3
"""
Example script for running floating feature selection.
"""

import sys
import os
import json
import pickle
from datetime import datetime

print("Starting Floating Feature Selection Example Script")
# Add the package directory to Python path for direct import
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'package'))
sys.path.insert(0, package_dir)

# Import directly from the ffs module
import ffs

# Configuration
run_id = 0              # Fixed seed for reproducibility
dataset = "recist_pancancer_union_cosmic_card_7" #"madelon"  # Name of the dataset being used

#data_path = "synthetic"  # Use an internal synthetic dataset (for development purposes)
#data_path = "data/melanoma_batch_corrected_data_common_genes_cosmic_io.csv"

if dataset == "madelon":
    data_path = "data/madelon_like_dataset.csv"

if dataset == "mirna_nb":
    data_path = "data/mirna_nb_dataset.csv"

if dataset == "recist_pancancer_cosmic_card_7":
    data_path = "data/data_processed/recist_pancancer_cosmic_card_7.csv"

if dataset == "recist_pancancer_union_cosmic_card_7":
    data_path = "data/data_processed/recist_pancancer_union_cosmic_card_7.csv"

if dataset == "recist_melanoma_cosmic_card_4":
    data_path = "data/data_processed/recist_melanoma_cosmic_card_4.csv"

if dataset == "recist_melanoma_union_cosmic_card_4":
    data_path = "data/data_processed/recist_melanoma_union_cosmic_card_4.csv"


target_column =  "recist" #"target" #"recist"  # Specify the target column for real datasets

n_experiments = 50     # Number of experiments to run

all_results = {}
for i in range(n_experiments):
    print(f"Running experiment {i+1}/{n_experiments} with run_id={run_id}")
    # Run the experiment    
    ffs_instance = ffs.FloatingFeatureSelector(run_id=run_id, data_path=data_path, target_column=target_column)
    experiment_result = ffs_instance.run_ffs(n_feat=15)  # Specify number of features to select before the optimization step

    print("Experiment completed successfully!")

    print("Selected features:", experiment_result)

    all_results[i+1] = {"selected_features": experiment_result, "run_id": i+1}



print("All experiments completed!")
print(all_results)

print("Top selected features from all experiments:")
# Count frequency of each feature across all experiments
feature_counts = {}
for experiment in all_results.values():
    for feature in experiment["selected_features"]:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1

# Sort features by frequency (descending)
sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

# Print top k features (e.g., top 10)
k = 20
print(f"Top {k} most selected features:")
for i, (feature, count) in enumerate(sorted_features[:k], 1):
    print(f" {feature}: selected {count}/{n_experiments} times")

# Save results to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Save as JSON (human-readable)
json_file = os.path.join(results_dir, f"ffs_results_{dataset}_{n_experiments}.json")
try:
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to JSON: {json_file}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# Save as pickle (preserves Python objects)
pickle_file = os.path.join(results_dir, f"ffs_results_{dataset}_{n_experiments}.pkl")
try:
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to pickle: {pickle_file}")
except Exception as e:
    print(f"Error saving pickle: {e}")

print("Results successfully saved!")
