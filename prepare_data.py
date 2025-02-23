#!/usr/bin/env python
import os
import zipfile
import shutil
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import glob
import subprocess

def extract_subject_data(zip_path, extract_dir):
    """Extract a subject's zip file to the specified directory using 7zip."""
    try:
        # Create extraction directory
        os.makedirs(extract_dir, exist_ok=True)
        
        # First, test the archive
        test_result = subprocess.run(['7z', 't', zip_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        
        if test_result.returncode != 0:
            return False, f"Archive test failed: {test_result.stderr}"
        
        # Extract with 7zip, excluding macOS metadata
        result = subprocess.run(['7z', 'x', '-y', '-o' + extract_dir, zip_path],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        
        if result.returncode != 0:
            return False, f"7zip extraction failed: {result.stderr}"
        
        # Clean up macOS metadata files and directories
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            # Remove macOS metadata files
            for name in files:
                if name.startswith('._') or name == '.DS_Store':
                    os.remove(os.path.join(root, name))
            # Remove macOS metadata directories
            for name in dirs:
                if name == '__MACOSX':
                    shutil.rmtree(os.path.join(root, name))
        
        # Move files from nested directory if they exist
        subject_name = os.path.splitext(os.path.basename(zip_path))[0]
        nested_dir = os.path.join(extract_dir, subject_name)
        if os.path.exists(nested_dir):
            for item in os.listdir(nested_dir):
                src = os.path.join(nested_dir, item)
                dst = os.path.join(extract_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            shutil.rmtree(nested_dir)
        
        return True, "Successfully extracted"
    except Exception as e:
        return False, f"Error extracting: {str(e)}"

def validate_subject_data(subject_dir):
    """Validate that a subject directory contains the required files and data structure."""
    required_files = ['data_full.mat', 'TimeSeries.h5']
    
    # Check for required files
    for file in required_files:
        file_path = os.path.join(subject_dir, file)
        if not os.path.exists(file_path):
            return False, f"Missing required file: {file}"
    
    try:
        # Validate MAT file structure
        mat_data = loadmat(os.path.join(subject_dir, 'data_full.mat'))
        if 'data' not in mat_data or not isinstance(mat_data['data'], np.ndarray):
            return False, "Invalid MAT file structure: missing 'data' field"
        
        data_struct = mat_data['data'][0,0]
        required_fields = ['CellXYZ']
        for field in required_fields:
            if field not in data_struct.dtype.names:
                return False, f"Invalid MAT file structure: missing '{field}' field"
        
        # Validate HDF5 file structure
        with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
            if 'CellResp' not in f:
                return False, "Invalid HDF5 file structure: missing 'CellResp' dataset"
            
            # Check that CellResp has expected shape (timepoints, cells)
            cell_resp = f['CellResp']
            if len(cell_resp.shape) != 2:
                return False, f"Invalid CellResp shape: expected 2D array, got {len(cell_resp.shape)}D"
    
    except Exception as e:
        return False, f"Error validating data: {str(e)}"
    
    return True, "Data validation successful"

def prepare_all_subjects(jdataraw_dir="jdataraw", processed_dir="data/processed_subjects"):
    """Prepare all subject data by extracting zips and validating data."""
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all zip files and the unzipped subject_1 directory
    zip_files = glob.glob(os.path.join(jdataraw_dir, "subject_*.zip"))
    subject_dirs = [d for d in glob.glob(os.path.join(jdataraw_dir, "subject_*")) if os.path.isdir(d)]
    
    print(f"Found {len(zip_files)} zip files and {len(subject_dirs)} unzipped directories")
    
    # Track extraction results
    extraction_results = []
    
    # Process each zip file
    for zip_file in tqdm(zip_files, desc="Extracting subjects"):
        subject_name = os.path.splitext(os.path.basename(zip_file))[0]
        extract_dir = os.path.join(processed_dir, subject_name)
        
        if os.path.exists(extract_dir):
            print(f"Skipping {subject_name}, already extracted")
            extraction_results.append((subject_name, True, "Already extracted"))
            continue
        
        print(f"\nExtracting {subject_name}...")
        success, message = extract_subject_data(zip_file, extract_dir)
        extraction_results.append((subject_name, success, message))
        
        if not success:
            print(f"Failed to extract {subject_name}: {message}")
            # Clean up failed extraction
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
    
    # Add unzipped directories if they're not already in processed_dir
    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)
        target_dir = os.path.join(processed_dir, subject_name)
        
        if not os.path.exists(target_dir):
            print(f"\nCopying {subject_name}...")
            try:
                shutil.copytree(subject_dir, target_dir)
                extraction_results.append((subject_name, True, "Copied from unzipped directory"))
            except Exception as e:
                print(f"Failed to copy {subject_name}: {str(e)}")
                extraction_results.append((subject_name, False, f"Copy failed: {str(e)}"))
    
    # Print extraction summary
    print("\nExtraction Summary:")
    print("-" * 60)
    for subject, success, message in extraction_results:
        status = "Success" if success else "Failed"
        print(f"{subject}: {status} - {message}")
    print("-" * 60)
    
    # Validate all processed subjects
    all_processed_dirs = glob.glob(os.path.join(processed_dir, "subject_*"))
    print("\nValidating all subjects...")
    
    valid_subjects = []
    invalid_subjects = []
    
    for subject_dir in tqdm(all_processed_dirs, desc="Validating subjects"):
        subject_name = os.path.basename(subject_dir)
        is_valid, message = validate_subject_data(subject_dir)
        
        if is_valid:
            valid_subjects.append(subject_name)
        else:
            invalid_subjects.append((subject_name, message))
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Total subjects processed: {len(all_processed_dirs)}")
    print(f"Valid subjects: {len(valid_subjects)}")
    print(f"Invalid subjects: {len(invalid_subjects)}")
    
    if invalid_subjects:
        print("\nInvalid subjects details:")
        for subject, message in invalid_subjects:
            print(f"{subject}: {message}")
    
    return valid_subjects, invalid_subjects

def get_dataset_stats(processed_dir="data/processed_subjects"):
    """Get statistics about the dataset across all valid subjects."""
    stats = {
        'total_cells': [],
        'valid_cells': [],
        'timepoints': [],
        'position_ranges': []
    }
    
    subject_dirs = glob.glob(os.path.join(processed_dir, "subject_*"))
    
    for subject_dir in tqdm(subject_dirs, desc="Computing dataset statistics"):
        try:
            # Load MAT file
            mat_data = loadmat(os.path.join(subject_dir, 'data_full.mat'))
            data_struct = mat_data['data'][0,0]
            
            # Get cell positions
            cell_xyz = data_struct['CellXYZ']
            if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
                cell_xyz = cell_xyz[0,0]
            
            total_cells = cell_xyz.shape[0]
            stats['total_cells'].append(total_cells)
            
            # Count valid cells
            if 'IX_inval_anat' in data_struct.dtype.names:
                invalid_indices = data_struct['IX_inval_anat']
                if isinstance(invalid_indices, np.ndarray) and invalid_indices.dtype == np.object_:
                    invalid_indices = invalid_indices[0,0].flatten()
                valid_cells = total_cells - len(invalid_indices)
            else:
                valid_cells = total_cells
            stats['valid_cells'].append(valid_cells)
            
            # Get position ranges
            stats['position_ranges'].append({
                'min': cell_xyz.min(axis=0),
                'max': cell_xyz.max(axis=0)
            })
            
            # Get timepoints from HDF5
            with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
                stats['timepoints'].append(f['CellResp'].shape[0])
        
        except Exception as e:
            print(f"Error processing {os.path.basename(subject_dir)}: {str(e)}")
            continue
    
    # Compute summary statistics
    if not stats['total_cells']:
        print("No valid subjects found!")
        return stats, {
            'num_subjects': 0,
            'avg_total_cells': 0,
            'avg_valid_cells': 0,
            'avg_timepoints': 0,
            'position_bounds': {'min': None, 'max': None}
        }
    
    summary = {
        'num_subjects': len(stats['total_cells']),
        'avg_total_cells': np.mean(stats['total_cells']),
        'avg_valid_cells': np.mean(stats['valid_cells']),
        'avg_timepoints': np.mean(stats['timepoints']),
        'position_bounds': {
            'min': np.min([r['min'] for r in stats['position_ranges']], axis=0),
            'max': np.max([r['max'] for r in stats['position_ranges']], axis=0)
        }
    }
    
    return stats, summary

if __name__ == "__main__":
    print("Starting data preparation...")
    valid_subjects, invalid_subjects = prepare_all_subjects()
    
    if valid_subjects:
        print("\nComputing dataset statistics...")
        stats, summary = get_dataset_stats()
        
        print("\nDataset Summary:")
        print(f"Number of valid subjects: {summary['num_subjects']}")
        print(f"Average total cells per subject: {summary['avg_total_cells']:.0f}")
        print(f"Average valid cells per subject: {summary['avg_valid_cells']:.0f}")
        print(f"Average timepoints per subject: {summary['avg_timepoints']:.0f}")
        print("\nPosition bounds across all subjects:")
        print(f"Min (x,y,z): {summary['position_bounds']['min']}")
        print(f"Max (x,y,z): {summary['position_bounds']['max']}") 