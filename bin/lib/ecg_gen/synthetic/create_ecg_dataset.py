#!/usr/bin/env python3
"""
Generate clinical ECG archetype dataset for federated learning research.
Creates realistic, non-iid ECG data distributed across multiple client archetypes.
"""

import os
import argparse
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

# Add the lib directory to Python path for imports
import sys
ROOT_DIR = "/gpfs/commons/groups/gursoy_lab/aelhussein/fl4tsf"
sys.path.append(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}/bin")
sys.path.append(f"{ROOT_DIR}/lib")
from lib.ecg_gen import generate_clients
from lib.dataset_utils import store_dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate clinical ECG archetype dataset for federated learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument("--dataset-prefix", type=str, default="ecg_clinical",
                       help="Prefix for dataset files")
    
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for datasets")
    
    # Client and patient parameters  
    parser.add_argument("--n-clients", type=int, default=10,
                       help="Number of federated clients")
    parser.add_argument("--n-patients", type=int, default=20,
                       help="Number of patients per client")
    
    # ECG parameters
    parser.add_argument("--duration", type=float, default=10.0,
                       help="ECG recording duration in seconds")
    
    # Distribution parameters
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                       help="Dirichlet concentration parameter (lower = more specialized clients)")
    
    parser.add_argument("--variable-duration", action="store_false", default=True,
                   help="Use fixed or variable recording durations (default: True, meaning variable durations are used)")
    
    # Technical parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Fraction of data to use for training")
    
    return parser.parse_args()


def collect_client_data(generator, n_clients):
    """Collect patient data from generator and group by client."""
    client_datasets = defaultdict(lambda: {
        'signals': [],
        'timestamps': [], 
        'metadata': []
    })
    
    total_patients = 0
    patients_per_client = defaultdict(int)
    
    print("Generating patient data...")
    for patient_data in generator:
        client_id = patient_data['metadata']['client_id']
        
        client_datasets[client_id]['signals'].append(patient_data['signal'])
        client_datasets[client_id]['timestamps'].append(patient_data['timestamps'])
        client_datasets[client_id]['metadata'].append(patient_data['metadata'])
        
        patients_per_client[client_id] += 1
        total_patients += 1
        
        if total_patients % 50 == 0:
            print(f"Generated {total_patients} patients...")
    
    print(f"Generated {total_patients} total patients across {len(client_datasets)} clients")
    
    # Print client statistics
    for client_id in sorted(client_datasets.keys()):
        archetype_counts = {}
        medication_counts = {}
        comorbidity_counts = {}
        geography_info = None
        
        for meta in client_datasets[client_id]['metadata']:
            archetype = meta['archetype']
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
            
            # Track geography (same for all patients in a client)
            if geography_info is None:
                geography_info = meta.get('geography', 'unknown')
            
            # Count medication usage
            for med in meta.get('medications', []):
                medication_counts[med] = medication_counts.get(med, 0) + 1
                
            # Count comorbidity prevalence
            for comorbidity in meta.get('comorbidity', []):
                comorbidity_counts[comorbidity] = comorbidity_counts.get(comorbidity, 0) + 1
        
        # Create summary strings (limit to top 3 for readability)
        med_summary = f", meds: {dict(list(medication_counts.items())[:3])}" if medication_counts else ""
        comorbid_summary = f", comorbidity: {dict(list(comorbidity_counts.items())[:3])}" if comorbidity_counts else ""
        geo_summary = f", region: {geography_info}" if geography_info != 'unknown' else ""
        
        print(f"Client {client_id}: {patients_per_client[client_id]} patients, "
              f"archetypes: {dict(archetype_counts)}{med_summary}{comorbid_summary}{geo_summary}")
    
    return client_datasets


def save_client_dataset(client_id, client_data, args):
    """Save dataset for a single client with support for variable-length signals."""
    dataset_path = ROOT_DIR / Path(args.output_dir) / f"{args.dataset_prefix}_{args.dirichlet_alpha}"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    client_prefix = f"client_{client_id}"
    
    # Handle variable-length signals by padding to max length
    signals = client_data['signals']
    timestamps = client_data['timestamps']
    
    # Find the maximum length across all signals
    max_length = max(signal.shape[0] for signal in signals)
    print(f"Signal lengths: {[signal.shape[0] for signal in signals]}")
    print(f"Max length: {max_length}")
    
    # Pad all signals to the same length
    padded_signals = []
    padded_timestamps = []
    
    for signal, timestamp in zip(signals, timestamps):
        signal_length = signal.shape[0]
        
        if signal_length < max_length:
            # Pad signal with zeros
            pad_length = max_length - signal_length
            if len(signal.shape) == 2:  # [length, channels]
                padded_signal = F.pad(signal, (0, 0, 0, pad_length), mode='constant', value=0)
            else:  # [length]
                padded_signal = F.pad(signal, (0, pad_length), mode='constant', value=0)
            
            # Pad timestamps - extend with the last timestamp + incremental steps
            if len(timestamp.shape) == 1:
                last_time = timestamp[-1]
                time_step = timestamp[1] - timestamp[0] if len(timestamp) > 1 else 1.0
                additional_times = torch.arange(1, pad_length + 1) * time_step + last_time
                padded_timestamp = torch.cat([timestamp, additional_times])
            else:
                # Handle multi-dimensional timestamps
                padded_timestamp = F.pad(timestamp, (0, 0, 0, pad_length), mode='constant', value=0)
        else:
            padded_signal = signal
            padded_timestamp = timestamp
            
        padded_signals.append(padded_signal)
        padded_timestamps.append(padded_timestamp)
    
    # Now stack the padded tensors
    try:
        signals_tensor = torch.stack(padded_signals)
        timestamps_tensor = torch.stack(padded_timestamps)
    except RuntimeError as e:
        print(f"Error stacking tensors after padding: {e}")
        print(f"Padded signal shapes: {[s.shape for s in padded_signals]}")
        print(f"Padded timestamp shapes: {[t.shape for t in padded_timestamps]}")
        raise
    
    # Create metadata DataFrame BEFORE reordering
    metadata_df = pd.DataFrame(client_data['metadata'])
    
    # Create train/test split
    n_patients = len(signals_tensor)
    split_idx = int(n_patients * args.train_split)
    indices = torch.randperm(n_patients)  # Shuffle for random split
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Reorder metadata to match tensor reordering
    all_indices = torch.cat([train_indices, test_indices])
    metadata_df_reordered = metadata_df.iloc[all_indices.tolist()].reset_index(drop=True)
    
    # Create split labels array and assign to reordered metadata
    split_labels = ['train'] * len(train_indices) + ['test'] * len(test_indices)
    metadata_df_reordered['split'] = split_labels
    
    # Debug: Print split distribution
    print(f"Split distribution: {metadata_df_reordered['split'].value_counts().to_dict()}")
    
    # Prepare data in expected format for store_dataset
    # Reorder tensors to match metadata order: [train_samples, test_samples]
    reordered_signals = torch.cat([signals_tensor[train_indices], signals_tensor[test_indices]])
    reordered_timestamps = torch.cat([timestamps_tensor[train_indices], timestamps_tensor[test_indices]])
    
    train_set = {
        'data': reordered_signals[:len(train_indices)],
        'time_steps': reordered_timestamps[:len(train_indices)],
        'amplitude': [],  # Not used for ECG data
        'frequency': []   # Not used for ECG data
    }
    
    test_set = {
        'data': reordered_signals[len(train_indices):],
        'time_steps': reordered_timestamps[len(train_indices):], 
        'amplitude': [],
        'frequency': []
    }

    # Save using existing store_dataset function
    store_dataset(
        train_set, 
        test_set, 
        dataset_prefix=f"{args.dataset_prefix}_{args.dirichlet_alpha}",
        path_prefix=f"{ROOT_DIR}/{args.output_dir}",
        client_prefix=client_prefix
    )
    
    # Add padding information to reordered metadata (use reordered indices)
    original_lengths = [client_data['signals'][i].shape[0] for i in all_indices.tolist()]
    metadata_df_reordered['original_length'] = original_lengths
    metadata_df_reordered['padded_length'] = max_length
    metadata_df_reordered['padding_applied'] = [length < max_length for length in original_lengths]
    
    # Save reordered metadata
    metadata_path = dataset_path / f"{client_prefix}_metadata.csv"
    metadata_df_reordered.to_csv(metadata_path, index=False)
    
    # Validate alignment between saved tensors and metadata
    print(f"Validating data alignment for {client_prefix}...")
    print(f"Total metadata rows: {len(metadata_df_reordered)}")
    print(f"Train tensor size: {len(train_set['data'])}, Test tensor size: {len(test_set['data'])}")
    
    train_metadata = metadata_df_reordered[metadata_df_reordered['split'] == 'train'].reset_index(drop=True)
    test_metadata = metadata_df_reordered[metadata_df_reordered['split'] == 'test'].reset_index(drop=True)
    
    print(f"Train metadata size: {len(train_metadata)}, Test metadata size: {len(test_metadata)}")
    
    # Check that the number of samples matches
    assert len(train_metadata) == len(train_set['data']), f"Train metadata ({len(train_metadata)}) and data ({len(train_set['data'])}) size mismatch"
    assert len(test_metadata) == len(test_set['data']), f"Test metadata ({len(test_metadata)}) and data ({len(test_set['data'])}) size mismatch"
    
    print(f"Saved {client_prefix}: {len(train_indices)} train, {len(test_indices)} test samples")
    print(f"Signal lengths - Original: {min(original_lengths)}-{max(original_lengths)}, Padded: {max_length}")
    print(f"Data alignment validated successfully!")

def save_dataset_summary(client_datasets, args):
    """Save overall dataset summary statistics."""
    dataset_path = ROOT_DIR / Path(args.output_dir) / f"{args.dataset_prefix}_{args.dirichlet_alpha}"
    
    # Collect summary statistics
    summary_data = []
    for client_id, data in client_datasets.items():
        archetype_counts = {}
        hr_values = []
        hrv_values = []
        qt_values = []
        medication_usage = {}
        comorbidity_usage = {}
        geography_info = None
        
        for meta in data['metadata']:
            archetype = meta['archetype']
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
            hr_values.append(meta['heart_rate'])
            hrv_values.append(meta['heart_rate_std'])
            qt_values.append(meta.get('qt_target_ms', 400))
            
            # Track geography (same for all patients in a client)
            if geography_info is None:
                geography_info = meta.get('geography', 'unknown')
            
            # Count medication usage
            for med in meta.get('medications', []):
                medication_usage[med] = medication_usage.get(med, 0) + 1
                
            # Count comorbidity usage
            for comorbidity in meta.get('comorbidity', []):
                comorbidity_usage[comorbidity] = comorbidity_usage.get(comorbidity, 0) + 1
        
        summary_data.append({
            'client_id': client_id,
            'n_patients': len(data['metadata']),
            'geography': geography_info,
            'primary_archetype': max(archetype_counts.keys(), key=archetype_counts.get),
            'archetype_counts': str(archetype_counts),
            'medication_usage': str(medication_usage),
            'comorbidity_usage': str(comorbidity_usage),
            'mean_hr': sum(hr_values) / len(hr_values),
            'mean_hrv_bpm': sum(hrv_values) / len(hrv_values),
            'mean_qt_ms': sum(qt_values) / len(qt_values),
            'hr_std': torch.tensor(hr_values).std().item(),
            'hrv_std': torch.tensor(hrv_values).std().item(),
            'qt_std': torch.tensor(qt_values).std().item()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = dataset_path / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Saved dataset summary to {summary_path}")
    
    # Print overall statistics
    total_patients = sum(len(data['metadata']) for data in client_datasets.values())
    geography_counts = summary_df['geography'].value_counts().to_dict()
    
    print(f"\nDataset Statistics:")
    print(f"Total patients: {total_patients}")
    print(f"Clients: {len(client_datasets)}")
    print(f"Geography distribution: {geography_counts}")
    print(f"Mean HR range: {summary_df['mean_hr'].min():.1f}-{summary_df['mean_hr'].max():.1f} bpm")
    print(f"Mean QT range: {summary_df['mean_qt_ms'].min():.0f}-{summary_df['mean_qt_ms'].max():.0f} ms")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print(f"Creating ECG dataset with {args.n_clients} clients, {args.n_patients} patients each")
    print(f"Duration: {args.duration}s, Variable duration: {args.variable_duration}, Alpha: {args.dirichlet_alpha}, Seed: {args.seed}")
    
    # Generate patient data
    data_generator = generate_clients(
        num_clients=args.n_clients,
        num_patients_per_client=args.n_patients,
        duration_sec=args.duration,
        variable_duration=args.variable_duration,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed
    )
    
    # Collect data by client
    client_datasets = collect_client_data(data_generator, args.n_clients)
    
    print(f"\nSaving datasets to {ROOT_DIR} / {args.output_dir}/{args.dataset_prefix}_{args.dirichlet_alpha}/...")
    
    # Save each client's dataset
    for client_id, client_data in client_datasets.items():
        save_client_dataset(client_id, client_data, args)
    
    # Save summary statistics
    save_dataset_summary(client_datasets, args)
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()