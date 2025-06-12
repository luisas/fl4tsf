#!/usr/bin/env python
import os
import torch
from torch.utils.data import ConcatDataset
import argparse
import re

def is_valid_filename(filename, data_type):
    pattern = fr"client_\d+_{data_type}\.pt"
    return re.fullmatch(pattern, filename) is not None

def merge_datasets(data_type, prefix='merged'):
    datasets = []
    for filename in os.listdir('.'):
        print(f'Checking file: {filename}')
        # keep files that have name client_{a number}_{data_type}.pt and nothing ekse, use regex
        if is_valid_filename(filename, data_type):
            dataset = torch.load(filename, weights_only=True)
            datasets.append(dataset)
    if datasets:
        merged_dataset = ConcatDataset(datasets)
        torch.save(merged_dataset, f'{prefix}_{data_type}.pt')
        print(f'Merged {len(datasets)} {data_type} datasets into {prefix}_{data_type}.pt')
    else:
        print(f'No datasets found for {data_type}.')
    
def main():
    
    parser = argparse.ArgumentParser(description='Merge train and test datasets.')
    parser.add_argument('--prefix', type=str, default='merged', help='Prefix for merged dataset files')
    args = parser.parse_args()

    print(f'Merging datasets with prefix: {args.prefix}')
    
    merge_datasets('train', args.prefix)
    merge_datasets('test', args.prefix)
    merge_datasets('time_steps_train', args.prefix)
    merge_datasets('time_steps_test', args.prefix)

if __name__ == "__main__":
    main()