#!/usr/bin/env python

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_data_min_max(records: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get minimum and maximum for each feature across the whole dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_min, data_max = None, None
    inf = torch.tensor(float("Inf"), device=device)

    for record_id, tt, vals, mask, labels in records:
        n_features = vals.size(-1)
        batch_min, batch_max = [], []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min_tensor = torch.stack(batch_min)
        batch_max_tensor = torch.stack(batch_max)

        if data_min is None:
            data_min, data_max = batch_min_tensor, batch_max_tensor
        else:
            data_min = torch.min(data_min, batch_min_tensor)
            data_max = torch.max(data_max, batch_max_tensor)

    return data_min, data_max

def save_federated_physionet_split(client_id: int, train_list: list, test_list: list, 
                                   data_min: torch.Tensor, data_max: torch.Tensor,
                                   metadata_df: pd.DataFrame, dataset_name: str, output_dir: str):
    """Save client data in PhysioNet-compatible format for FL pipeline."""
    save_path = Path(output_dir) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    client_prefix = f"client_{client_id}"
    output_file_prefix = save_path / client_prefix

    torch.save(train_list, f"{output_file_prefix}_train.pt")
    torch.save(test_list, f"{output_file_prefix}_test.pt")
    torch.save(data_min, f"{output_file_prefix}_data_min.pt")
    torch.save(data_max, f"{output_file_prefix}_data_max.pt")
    metadata_df.to_csv(f"{output_file_prefix}_metadata.csv", index=False)
    
    print(f"✅ Saved PhysioNet-style data for client {client_id} to '{save_path}'")

def load_and_preprocess_data(data_path: str, scp_statements_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and preprocess data from .npy file."""
    print(f"Loading data from: {data_path}")
    
    # Load from .npy file
    data = np.load(data_path, allow_pickle=True).item()
    Y = data['metadata']
    X_raw = data['raw']
    X_emb = data['emb']
    
    print(f"Original shapes - Y: {Y.shape}, X_raw: {X_raw.shape}, X_emb: {X_emb.shape}")
    
    # Store original indices to track alignment
    original_indices = Y.index.copy()
    
    # Filter Y - step 1: dropna
    Y = Y.dropna(subset=['scp_codes', 'site'])
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(scp_statements_path, index_col=0)
    agg_df = agg_df[(agg_df.diagnostic == 1) | (agg_df.index.isin(['AFIB', 'PACE', 'AFLT']))]
    agg_df['diagnostic_class'] = agg_df['diagnostic_class'].fillna(agg_df.index.to_series())
    
    # Create fast lookup dictionary
    code_to_class = agg_df['diagnostic_class'].to_dict()
    valid_codes = set(code_to_class.keys())
    all_diagnostic_classes = sorted(agg_df['diagnostic_class'].unique())

    def fast_aggregate_diagnostic_wide(y_dic):
        """Return a dictionary with binary indicators for each diagnostic class."""
        if not isinstance(y_dic, dict) or pd.isna(y_dic):
            return {cls: 0 for cls in all_diagnostic_classes}
        
        # Filter out codes with score < 50
        filtered_dic = {k: v for k, v in y_dic.items() if v >= 50}
        
        # If 'NORM' is present and there are other codes, drop 'NORM'
        if 'NORM' in filtered_dic and len(filtered_dic) > 1:
            del filtered_dic['NORM']
        
        # Map to diagnostic classes
        matching_codes = set(filtered_dic.keys()) & valid_codes
        present_classes = {code_to_class[code] for code in matching_codes}
        
        return {cls: 1 if cls in present_classes else 0 for cls in all_diagnostic_classes}

    # Apply the function and convert to DataFrame
    diagnostic_wide = Y.scp_codes.apply(fast_aggregate_diagnostic_wide)
    diagnostic_df = pd.DataFrame(diagnostic_wide.tolist(), index=Y.index)
    diagnostic_df.columns = [f'diag_{col}' for col in diagnostic_df.columns]
    Y_with_diagnostics = pd.concat([Y, diagnostic_df], axis=1)

    # Filter out rows with no diagnostics
    diagnostic_cols = [f'diag_{col}' for col in all_diagnostic_classes]
    Y_with_diagnostics = Y_with_diagnostics[Y_with_diagnostics[diagnostic_cols].sum(axis=1) > 0]
    
    print(f"Filtered Y shape: {Y_with_diagnostics.shape}")

    # Filter X_raw and X_emb to match filtered Y
    kept_indices = Y_with_diagnostics.index
    original_positions = pd.Series(range(len(original_indices)), index=original_indices)
    array_positions = original_positions.loc[kept_indices].values

    X_raw_filtered = X_raw[array_positions]
    X_emb_filtered = X_emb[array_positions]
    
    print(f"Filtered shapes - Y: {Y_with_diagnostics.shape}, X_raw: {X_raw_filtered.shape}, X_emb: {X_emb_filtered.shape}")
    
    return Y_with_diagnostics, X_raw_filtered, X_emb_filtered

class FederatedDataSplitter:
    """Create different levels of data splits for federated learning."""
    
    def __init__(self, df: pd.DataFrame, X_raw: np.ndarray, X_emb: np.ndarray = None):
        self.df = df.copy()
        if not self.df.index.is_unique:
            self.df = self.df.reset_index(drop=True)
            
        self.X_raw = X_raw.copy()
        self.X_emb = X_emb.copy() if X_emb is not None else None
        
        assert len(self.df) == self.X_raw.shape[0], "Y and X_raw lengths must match"
        
        self.diagnostic_cols = [col for col in df.columns if col.startswith('diag_')]
        self._create_superclasses()
        
        print(f"Initialized FederatedDataSplitter with {len(self.df)} samples.")

    def _create_superclasses(self):
        """Create a single 'superclass' column for stratification."""
        def get_superclass(row):
            if row.get('diag_NORM', 0) == 1: return 'NORM'
            if row.get('diag_MI', 0) == 1: return 'MI'
            if row.get('diag_STTC', 0) == 1: return 'STTC'
            if row.get('diag_CD', 0) == 1: return 'CD'
            if row.get('diag_HYP', 0) == 1: return 'HYP'
            if row.get('diag_AFIB', 0) == 1 or row.get('diag_AFLT', 0) == 1: return 'AFIB_AFLT'
            if row.get('diag_PACE', 0) == 1: return 'PACE'
            # Find first positive diagnosis
            for col in self.diagnostic_cols:
                if row.get(col, 0) == 1:
                    return col.replace('diag_', '')
            return 'UNKNOWN'
        
        self.df['superclass'] = self.df.apply(get_superclass, axis=1)
        print(f"Superclass distribution:")
        print(self.df['superclass'].value_counts())

    def _split_features(self, client_y_df: pd.DataFrame):
        """Get feature data corresponding to a metadata DataFrame."""
        client_indices = self.df.index.get_indexer(client_y_df.index)
        client_features = {'X_raw': self.X_raw[client_indices]}
        if self.X_emb is not None:
            client_features['X_emb'] = self.X_emb[client_indices]
        return client_features

    def _pool_data_from_sites(self, sites: List[int] = [0, 1]) -> pd.DataFrame:
        pooled_data = self.df[self.df['site'].isin(sites)].copy()
        print(f"Pooled {len(pooled_data)} samples from sites {sites}")
        return pooled_data

    def _print_distribution_summary(self, client_1_data: pd.DataFrame, client_2_data: pd.DataFrame, split_name: str):
        print(f"\n{split_name} - Superclass Distribution:")
        print("Client 1:")
        print(client_1_data['superclass'].value_counts(normalize=True).apply("{:.2%}".format))
        print("\nClient 2:")
        print(client_2_data['superclass'].value_counts(normalize=True).apply("{:.2%}".format))
        print("-" * 50)

    def level_0_iid_baseline(self, sites: List[int] = [0, 1], random_state: int = 42):
        """Level 0: IID Baseline - Purely random split."""
        print("\n=== LEVEL 0: IID BASELINE ===")
        pooled_data = self._pool_data_from_sites(sites)
        shuffled_data = shuffle(pooled_data, random_state=random_state)
        mid_point = len(shuffled_data) // 2
        client_1_data = shuffled_data.iloc[:mid_point].copy()
        client_2_data = shuffled_data.iloc[mid_point:].copy()
        c1_feat, c2_feat = self._split_features(client_1_data), self._split_features(client_2_data)
        self._print_distribution_summary(client_1_data, client_2_data, "IID Baseline")
        return client_1_data, client_2_data, c1_feat, c2_feat

    def _dirichlet_split(self, sites: List[int], alpha: float, random_state: int, split_name: str):
        """Split data using Dirichlet distribution for label skew."""
        np.random.seed(random_state)
        pooled_data = self._pool_data_from_sites(sites)
        superclasses = pooled_data['superclass'].unique()
        n_classes, n_clients = len(superclasses), 2
        class_distributions = {sc: np.random.dirichlet([alpha] * n_clients) for sc in superclasses}
        
        client_1_indices, client_2_indices = [], []
        for superclass in superclasses:
            class_data = pooled_data[pooled_data['superclass'] == superclass]
            class_indices = class_data.index.to_numpy()
            np.random.shuffle(class_indices)
            
            n_client_1 = int(len(class_indices) * class_distributions[superclass][0])
            client_1_indices.extend(class_indices[:n_client_1])
            client_2_indices.extend(class_indices[n_client_1:])

        client_1_data = self.df.loc[client_1_indices].copy()
        client_2_data = self.df.loc[client_2_indices].copy()

        c1_feat, c2_feat = self._split_features(client_1_data), self._split_features(client_2_data)
        self._print_distribution_summary(client_1_data, client_2_data, split_name)
        return client_1_data, client_2_data, c1_feat, c2_feat

    def level_1_mild_label_skew(self, alpha: float = 1.0, sites: List[int] = [0, 1], random_state: int = 42):
        """Level 1: Mild Label Skew using Dirichlet distribution."""
        print(f"\n=== LEVEL 1: MILD LABEL SKEW (alpha={alpha}) ===")
        return self._dirichlet_split(sites, alpha, random_state, "Mild Label Skew")

    def level_2_severe_label_skew(self, alpha: float = 0.1, sites: List[int] = [0, 1], random_state: int = 42):
        """Level 2: Severe Label Skew using Dirichlet distribution."""
        print(f"\n=== LEVEL 2: SEVERE LABEL SKEW (alpha={alpha}) ===")
        return self._dirichlet_split(sites, alpha, random_state, "Severe Label Skew")

    def level_3_natural_feature_shift(self, target_size: int = None, random_state: int = 42):
        """Level 3: Natural Feature Shift (site-based split)."""
        print("\n=== LEVEL 3: NATURAL FEATURE SHIFT (SITE-BASED SPLIT) ===")
        
        np.random.seed(random_state)
        site_0_data = self.df[self.df['site'] == 0].copy()
        site_1_data = self.df[self.df['site'] == 1].copy()
        
        print(f"Site 0: {len(site_0_data)} samples")
        print(f"Site 1: {len(site_1_data)} samples")
        
        client_1_data, client_2_data = site_0_data, site_1_data
        
        if target_size:
            if len(client_1_data) > target_size:
                client_1_data = client_1_data.sample(n=target_size, random_state=random_state)
            if len(client_2_data) > target_size:
                client_2_data = client_2_data.sample(n=target_size, random_state=random_state + 1)
        
        c1_feat = self._split_features(client_1_data)
        c2_feat = self._split_features(client_2_data)
        
        self._print_distribution_summary(client_1_data, client_2_data, "Natural Feature Shift")
        print(f"Final sizes - Client 1: {len(client_1_data)}, Client 2: {len(client_2_data)}")
        return client_1_data, client_2_data, c1_feat, c2_feat

    def _apply_label_skew(self, site_data, target_dist, target_size, random_state):
        """Apply controlled label skew to site data."""
        np.random.seed(random_state)
        target_counts = {k: int(v * target_size) for k, v in target_dist.items()}
        sampled_data = []
        for class_name, count in target_counts.items():
            if class_name == 'OTHER':
                # Handle 'OTHER' as all classes not explicitly mentioned
                other_classes = [cls for cls in self.df['superclass'].unique() 
                               if cls not in target_dist.keys() or cls == 'OTHER']
                class_df = site_data[site_data['superclass'].isin(other_classes)]
            else:
                class_df = site_data[site_data['superclass'] == class_name]
            
            if len(class_df) >= count:
                sampled_data.append(class_df.sample(n=count, random_state=random_state))
            else:
                sampled_data.append(class_df)
        return pd.concat(sampled_data) if sampled_data else pd.DataFrame()

    def level_4_extreme_combined_non_iid(self, target_size: int = 1000, random_state: int = 42,
                                         client_1_target_dist: Dict = None, client_2_target_dist: Dict = None):
        """Level 4: Extreme Combined Non-IID (site + controlled label skew)."""
        print("\n=== LEVEL 4: EXTREME COMBINED NON-IID ===")
        
        if client_1_target_dist is None: 
            client_1_target_dist = {'MI': 0.4, 'CD': 0.4, 'OTHER': 0.2}
        if client_2_target_dist is None: 
            client_2_target_dist = {'NORM': 0.8, 'OTHER': 0.2}

        site_0_data = self.df[self.df['site'] == 0].copy()
        site_1_data = self.df[self.df['site'] == 1].copy()

        client_1_data = self._apply_label_skew(site_0_data, client_1_target_dist, target_size, random_state)
        client_2_data = self._apply_label_skew(site_1_data, client_2_target_dist, target_size, random_state + 1)

        c1_feat, c2_feat = self._split_features(client_1_data), self._split_features(client_2_data)
        self._print_distribution_summary(client_1_data, client_2_data, "Extreme Combined Non-IID")
        return client_1_data, client_2_data, c1_feat, c2_feat

    def level_5_site_temporal_split(self, target_size: int = None, random_state: int = 42,
                                   client_1_date_range: Tuple = ('1995-01-01', None),
                                   client_2_date_range: Tuple = (None, '1994-01-01')):
        """Level 5: Site + Temporal Split."""
        print("\n=== LEVEL 5: SITE + TEMPORAL SPLIT ===")
        np.random.seed(random_state)
        
        def filter_by_date(df, date_range):
            start, end = date_range
            if start: 
                start = pd.to_datetime(start)
                df = df[df['recording_date'] >= start]
            if end: 
                end = pd.to_datetime(end)
                df = df[df['recording_date'] < end]
            return df

        site_0_data = filter_by_date(self.df[self.df['site'] == 0].copy(), client_1_date_range)
        site_1_data = filter_by_date(self.df[self.df['site'] == 1].copy(), client_2_date_range)
        
        client_1_data, client_2_data = site_0_data, site_1_data
        
        if target_size:
            if len(client_1_data) > target_size:
                client_1_data = client_1_data.sample(n=target_size, random_state=random_state)
            if len(client_2_data) > target_size:
                client_2_data = client_2_data.sample(n=target_size, random_state=random_state + 1)

        c1_feat, c2_feat = self._split_features(client_1_data), self._split_features(client_2_data)
        self._print_distribution_summary(client_1_data, client_2_data, "Site + Temporal Split")
        return client_1_data, client_2_data, c1_feat, c2_feat

    def save_split(self, client_1_data: pd.DataFrame, client_2_data: pd.DataFrame,
                   client_1_features: Dict, client_2_features: Dict,
                   dataset_name: str, output_dir: str, train_frac: float = 0.8):
        """Save split in PhysioNet format."""
        print(f"\nSaving split for dataset '{dataset_name}' to '{output_dir}'...")
        self._save_client_data(0, client_1_data, client_1_features['X_raw'], dataset_name, output_dir, train_frac)
        self._save_client_data(1, client_2_data, client_2_features['X_raw'], dataset_name, output_dir, train_frac)
        print("Dataset successfully saved.")

    def _save_client_data(self, client_id: int, client_y: pd.DataFrame, client_x: np.ndarray,
                         dataset_name: str, output_dir: str, train_frac: float):
        """Save individual client data."""
        # Try stratified split first, fall back to non-stratified if needed
        try:
            train_y, test_y, train_x, test_x = train_test_split(
                client_y, client_x, train_size=train_frac, random_state=42, 
                stratify=client_y['superclass']
            )
            print(f"Client {client_id}: {len(train_y)} train, {len(test_y)} test samples (stratified).")
        except ValueError as e:
            if "least populated class" in str(e):
                # Fall back to non-stratified split for very non-IID scenarios
                print(f"⚠️  Client {client_id}: Some classes have too few samples for stratified split, using random split")
                train_y, test_y, train_x, test_x = train_test_split(
                    client_y, client_x, train_size=train_frac, random_state=42
                )
                print(f"Client {client_id}: {len(train_y)} train, {len(test_y)} test samples (non-stratified).")
            else:
                raise e

        def convert_to_list_format(data_x, data_y, sampling_rate=100):
            """Convert to PhysioNet format: list of (record_id, timesteps, vals, mask, labels)."""
            output_list = []
            timesteps = torch.linspace(0, (data_x.shape[1] - 1) / sampling_rate, data_x.shape[1])
            
            for i in range(len(data_x)):
                record_id = data_y.index[i]
                vals = torch.tensor(data_x[i], dtype=torch.float32)
                mask = torch.ones_like(vals)
                output_list.append((record_id, timesteps.clone(), vals, mask, None))
            return output_list

        train_list = convert_to_list_format(train_x, train_y)
        test_list = convert_to_list_format(test_x, test_y)
        
        # Get normalization stats
        data_min, data_max = get_data_min_max(train_list)
        
        # Save data
        save_federated_physionet_split(client_id, train_list, test_list, data_min, data_max, 
                                       client_y, dataset_name, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Create federated splits from PTB-XL data.")
    parser.add_argument("--dataset_prefix", type=str, default="ecg_physionet",
                        help="Prefix for output datasets (must contain 'physionet')")
    parser.add_argument("--data_path", type=str, 
                        default="/gpfs/commons/groups/gursoy_lab/aelhussein/fl4tsf/data/ecg_physionet/data_raw_emb_large.npy",
                        help="Path to .npy data file")
    parser.add_argument("--scp_statements_path", type=str,
                        default="/gpfs/commons/groups/gursoy_lab/aelhussein/fl4tsf/data/ecg_physionet/scp_statements.csv",
                        help="Path to scp_statements.csv")
    parser.add_argument("--output_dir", type=str, 
                        default="/gpfs/commons/groups/gursoy_lab/aelhussein/fl4tsf/data/ecg_physionet/splits",
                        help="Output directory")
    parser.add_argument("--target_size", type=int, default=500,
                        help="Target samples per client")
    
    args = parser.parse_args()

    if "physionet" not in args.dataset_prefix:
        raise ValueError("Dataset prefix must contain 'physionet' for pipeline compatibility")

    # Load and preprocess data
    Y_df, X_raw, X_emb = load_and_preprocess_data(args.data_path, args.scp_statements_path)
    
    # Initialize splitter
    splitter = FederatedDataSplitter(Y_df, X_raw, X_emb)
    
    # Generate all splits
    print("\n" + "="*80)
    print("GENERATING ALL FEDERATED SPLITS")
    print("="*80)
    
    # Level 0: IID Baseline
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_0_iid_baseline()
    dataset_name = f"{args.dataset_prefix}_level0_iid"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)

    # Level 1: Mild Label Skew
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_1_mild_label_skew(alpha=1.0)
    dataset_name = f"{args.dataset_prefix}_level1_mild_skew"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)

    # Level 2: Severe Label Skew
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_2_severe_label_skew(alpha=0.1)
    dataset_name = f"{args.dataset_prefix}_level2_severe_skew"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)

    # Level 3: Natural Feature Shift
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_3_natural_feature_shift(target_size=args.target_size)
    dataset_name = f"{args.dataset_prefix}_level3_natural"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)

    # Level 4: Extreme Combined Non-IID
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_4_extreme_combined_non_iid(target_size=args.target_size)
    dataset_name = f"{args.dataset_prefix}_level4_natural_and_severe_skew"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)

    # Level 5: Site + Temporal Split
    c1_y, c2_y, c1_feat, c2_feat = splitter.level_5_site_temporal_split(target_size=args.target_size)
    dataset_name = f"{args.dataset_prefix}_level5_temporal"
    splitter.save_split(c1_y, c2_y, c1_feat, c2_feat, dataset_name, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL SPLITS GENERATED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()