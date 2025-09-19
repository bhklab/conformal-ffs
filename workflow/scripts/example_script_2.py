

#!/usr/bin/env python3
"""
Example script for running floating feature selection.
"""

import sys
import os

# Add the package directory to Python path for direct import
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'package'))
sys.path.insert(0, package_dir)

# Import directly from the ffs module
import ffs

# Configuration
run_id = 0              # Fixed seed for reproducibility


#data_path = "synthetic"  # Use an internal synthetic dataset (for development purposes)
data_path = "data/melanoma_batch_corrected_data_common_genes_cosmic_io.csv"


# Run the experiment    
ffs_instance = ffs.FloatingFeatureSelector(run_id=run_id, data_path=data_path)
results = ffs_instance.run_ffs()

print("Experiment completed successfully!")