#!/usr/bin/env python3
"""
Script to find the epoch with maximum F1@50 score for each split and save all metrics to CSV.
"""

import os
import numpy as np
import re
from pathlib import Path

def extract_epoch_from_filename(filename):
    """Extract epoch number from filename like 'test_results_decoder-agg_epoch500.npy'"""
    match = re.search(r'epoch(\d+)\.npy$', filename)
    if match:
        return int(match.group(1))
    return None

def find_best_f1_epoch(base_path):
    """
    Find the epoch with maximum F1@50 score for each split and collect all metrics.
    
    Args:
        base_path (str): Path to the base directory containing splits
    
    Returns:
        list: List of dictionaries containing all metrics for best F1@50 epoch per split
    """
    base_path = Path(base_path)
    results = []
    
    # Iterate through each split directory
    for split_dir in sorted(base_path.glob('split_*')):
        if not split_dir.is_dir():
            continue
            
        split_name = split_dir.name
        print(f"\nProcessing {split_name}...")
        
        # Find the subdirectory containing the test results
        # Based on your structure, it seems to be JIGSAW-Trained-S*_M2R2_features
        model_dirs = list(split_dir.glob('*_BRP_features'))
        
        if not model_dirs:
            print(f"  No M2R2_features directory found in {split_name}")
            continue
            
        model_dir = model_dirs[0]  # Take the first one if multiple exist
        model_name = model_dir.name
        print(f"  Using model directory: {model_name}")
        
        best_epoch = None
        best_f1_score = -1
        best_metrics = None
        
        # Find all test_results files
        npy_files = list(model_dir.glob('test_results_decoder-agg_epoch*.npy'))
        
        if not npy_files:
            print(f"  No test results files found in {split_name}")
            continue
            
        print(f"  Found {len(npy_files)} epoch files")
        
        # Check each epoch file
        for npy_file in npy_files:
            epoch = extract_epoch_from_filename(npy_file.name)
            if epoch is None:
                print(f"  Could not extract epoch from {npy_file.name}")
                continue
                
            try:
                # Load the numpy file
                data = np.load(npy_file, allow_pickle=True)
                
                # Extract metrics
                # The data is a numpy array containing a dictionary
                if isinstance(data, np.ndarray) and data.size == 1:
                    metrics_dict = data.item()
                    if 'F1@50' in metrics_dict:
                        f1_50_score = float(metrics_dict['F1@50'])
                        
                        print(f"    Epoch {epoch:3d}: F1@50 = {f1_50_score:.4f}")
                        
                        if f1_50_score > best_f1_score:
                            best_f1_score = f1_50_score
                            best_epoch = epoch
                            # Store all metrics for this best epoch
                            best_metrics = metrics_dict.copy()
                    else:
                        print(f"    Epoch {epoch:3d}: F1@50 key not found")
                else:
                    print(f"    Epoch {epoch:3d}: Unexpected data format")
                    
            except Exception as e:
                print(f"    Error loading {npy_file.name}: {e}")
                continue
        
        if best_epoch is not None and best_metrics is not None:
            # Create a result dictionary with all information
            result_dict = {
                'Split': split_name,
                'Model': model_name,
                'Best_Epoch': best_epoch,
            }
            # Add all metrics from the best epoch
            for metric_name, metric_value in best_metrics.items():
                result_dict[metric_name] = metric_value
            
            results.append(result_dict)
            print(f"  Best epoch for {split_name}: {best_epoch} (F1@50 = {best_f1_score:.4f})")
            print(f"    All metrics: {best_metrics}")
        else:
            print(f"  No valid results found for {split_name}")
    
    return results

def main():
    # Check if pandas is available
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False
        print("Warning: pandas not found. Install with 'pip install pandas' for better CSV export.")
        print("Will continue with basic CSV export functionality.")
    
    # Set your base path here
    base_path = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/results/DiffAct/ImPerfectPour_BRP_features"
    
    # You can also pass it as a command line argument
    import sys
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    
    print(f"Searching for best F1@50 epochs in: {base_path}")
    
    # Find best epochs and collect all metrics
    results = find_best_f1_epoch(base_path)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Best F1@50 epochs for each split:")
    print("="*80)
    
    if results:
        if has_pandas:
            # Create DataFrame for better display and CSV export
            df = pd.DataFrame(results)
            
            # Display summary table
            print(f"{'Split':<15} {'Model':<30} {'Epoch':<6} {'F1@50':<8} {'Acc':<8} {'Edit':<8}")
            print("-" * 80)
            
            for _, row in df.iterrows():
                print(f"{row['Split']:<15} {row['Model']:<30} {row['Best_Epoch']:<6} "
                      f"{row['F1@50']:<8.2f} {row.get('Acc', 'N/A'):<8} {row.get('Edit', 'N/A'):<8}")
            
            # Create CSV file
            csv_filename = Path(base_path) / "best_f1_epochs_all_metrics.csv"
            df.to_csv(csv_filename, index=False, float_format='%.4f')
            print(f"\nDetailed results saved to CSV: {csv_filename}")
            
            # Display all available metrics
            metric_columns = [col for col in df.columns if col not in ['Split', 'Model', 'Best_Epoch']]
            print(f"\nAll available metrics: {', '.join(metric_columns)}")
            
        else:
            # Fallback display without pandas
            print(f"{'Split':<15} {'Model':<30} {'Epoch':<6} {'F1@50':<8}")
            print("-" * 65)
            
            for result in results:
                print(f"{result['Split']:<15} {result['Model']:<30} {result['Best_Epoch']:<6} "
                      f"{result['F1@50']:<8.2f}")
            
            # Manual CSV creation without pandas
            csv_filename = Path(base_path) / "best_f1_epochs_all_metrics.csv"
            if results:
                # Get all unique metric keys
                all_keys = set()
                for result in results:
                    all_keys.update(result.keys())
                
                # Sort keys for consistent column order
                ordered_keys = ['Split', 'Model', 'Best_Epoch'] + sorted([k for k in all_keys if k not in ['Split', 'Model', 'Best_Epoch']])
                
                with open(csv_filename, 'w') as f:
                    # Write header
                    f.write(','.join(ordered_keys) + '\n')
                    
                    # Write data rows
                    for result in results:
                        row_values = []
                        for key in ordered_keys:
                            value = result.get(key, '')
                            if isinstance(value, float):
                                row_values.append(f"{value:.4f}")
                            else:
                                row_values.append(str(value))
                        f.write(','.join(row_values) + '\n')
                
                print(f"\nDetailed results saved to CSV: {csv_filename}")
        
        # Create detailed text summary (works with or without pandas)
        summary_filename = Path(base_path) / "best_f1_epochs_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write("Best F1@50 epochs for each split with all metrics:\n")
            f.write("="*60 + "\n\n")
            
            for result in results:
                f.write(f"Split: {result['Split']}\n")
                f.write(f"Model: {result['Model']}\n")
                f.write(f"Best Epoch: {result['Best_Epoch']}\n")
                f.write("Metrics:\n")
                
                # Write all metrics except the metadata columns
                for key, value in result.items():
                    if key not in ['Split', 'Model', 'Best_Epoch']:
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n" + "-"*40 + "\n\n")
        
        print(f"Detailed summary saved to: {summary_filename}")
        
    else:
        print("No results found!")
    
    return results

if __name__ == "__main__":
    main()