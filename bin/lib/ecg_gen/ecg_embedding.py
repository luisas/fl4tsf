#!/usr/bin/env python3
"""
ECG Embedding Extraction using HuBERT-ECG Model

This script loads ECG data from the PTB-XL dataset, processes it, and extracts
embeddings using the HuBERT-ECG model from Hugging Face.

Requirements:
- torch
- transformers
- wfdb
- numpy
- pandas
- tqdm (optional, for progress bars)
"""

import os
import sys
import ast
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import wfdb
from transformers import AutoModel, AutoConfig
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ECGEmbeddingExtractor:
    """Extract ECG embeddings using HuBERT-ECG model."""
    
    def __init__(self, model_name: str = "Edoardo-BS/hubert-ecg-small"):
        """
        Initialize the ECG embedding extractor.
        
        Args:
            model_name: HuBERT-ECG model name from Hugging Face
        """
        self.model_name = model_name
        self.device = self._setup_device()
        self.model = None
        
        # Model parameters based on HuBERT-ECG paper
        self.INPUT_ECG_SECONDS = 5
        self.INPUT_SAMPLE_RATE_HZ = 100
        self.NUM_LEADS = 12
        self.SAMPLES_PER_LEAD_FOR_MODEL = self.INPUT_ECG_SECONDS * self.INPUT_SAMPLE_RATE_HZ
        self.FLATTENED_INPUT_LENGTH = self.NUM_LEADS * self.SAMPLES_PER_LEAD_FOR_MODEL
        
    def _setup_device(self) -> torch.device:
        """Setup CUDA or CPU device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU.")
        return device
    
    def load_model(self):
        """Load the HuBERT-ECG model."""
        logger.info(f"Loading model: {self.model_name}...")
        try:
            # Load model configuration
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Load model weights
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                config=config, 
                trust_remote_code=True
            )
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_raw_data(self, df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
        """
        Load raw ECG data using wfdb.
        
        Args:
            df: DataFrame containing ECG metadata
            sampling_rate: Sampling rate (100 or 500 Hz)
            path: Path to ECG files
            
        Returns:
            Array of ECG signals
        """
        logger.info(f"Loading raw ECG data with sampling rate: {sampling_rate} Hz")
        
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        
        data = []
        for filename in tqdm(df[filename_col], desc="Loading ECG files"):
            try:
                signal, meta = wfdb.rdsamp(f'{path}/{filename}')
                data.append(signal)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                # Add a zero array as placeholder
                data.append(np.zeros((1000, 12)))  # Default shape
                
        return np.array(data)
    
    def preprocess_ecg_data(self, X_raw: np.ndarray) -> torch.Tensor:
        """
        Preprocess ECG data for HuBERT-ECG model.
        
        Args:
            X_raw: Raw ECG data of shape (num_ecgs, time_samples, num_leads)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        logger.info(f"Preprocessing ECG data. Original shape: {X_raw.shape}")
        
        # Take first 5 seconds (500 samples at 100 Hz)
        X_processed_5s = X_raw[:, :self.SAMPLES_PER_LEAD_FOR_MODEL, :]
        
        # Flatten each ECG into 1D array
        X_flattened = X_processed_5s.reshape(-1, self.FLATTENED_INPUT_LENGTH)
        
        # Convert to tensor and move to device
        input_tensor = torch.from_numpy(X_flattened).float().to(self.device)
        
        logger.info(f"Preprocessed shape: {X_flattened.shape}")
        return input_tensor
    
    def extract_embeddings(self, input_tensor: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from preprocessed ECG data.
        
        Args:
            input_tensor: Preprocessed ECG tensor
            batch_size: Batch size for processing
            
        Returns:
            ECG embeddings array
        """
        logger.info("Extracting embeddings...")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embeddings_list = []
        num_samples = input_tensor.shape[0]
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
                batch = input_tensor[i:i+batch_size]
                
                try:
                    outputs = self.model(input_values=batch)
                    # Get fragment-level embeddings and average across fragments
                    fragment_embeddings = outputs.last_hidden_state
                    ecg_embeddings = fragment_embeddings.mean(dim=1)
                    embeddings_list.append(ecg_embeddings.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size}: {e}")
                    # Add zero embeddings as placeholder
                    hidden_size = 512  # Default for small model
                    batch_size_actual = batch.shape[0]
                    embeddings_list.append(np.zeros((batch_size_actual, hidden_size)))
        
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Final embeddings shape: {embeddings.shape}")
        return embeddings


def load_ptbxl_metadata(data_dir: str) -> pd.DataFrame:
    """Load and preprocess PTB-XL metadata."""
    logger.info("Loading PTB-XL metadata...")
    
    metadata_path = f"{data_dir}/ecg_physionet/ptbxl_database.csv"
    df = pd.read_csv(metadata_path, index_col='ecg_id')
    
    # Parse SCP codes
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Select relevant columns
    columns = [
        'patient_id', 'age', 'sex', 'height', 'weight', 'nurse', 'site',
        'device', 'recording_date', 'report', 'scp_codes', 'heart_axis',
        'infarction_stadium1', 'infarction_stadium2', 'baseline_drift', 
        'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 
        'pacemaker', 'filename_lr', 'filename_hr'
    ]
    
    df = df[columns]
    df['recording_date'] = pd.to_datetime(df['recording_date'])
    
    logger.info(f"Loaded metadata for {len(df)} ECG records")
    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract ECG embeddings using HuBERT-ECG")
    parser.add_argument("--root_dir", type=str, required=True,
                       help="Root directory containing the ECG data")
    parser.add_argument("--model_name", type=str, default="Edoardo-BS/hubert-ecg-small",
                       help="HuBERT-ECG model name")
    parser.add_argument("--sampling_rate", type=int, default=100,
                       help="ECG sampling rate (100 or 500 Hz)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Setup paths
    root_dir = Path(args.root_dir)
    data_dir = root_dir / "data"
    ecg_path = data_dir / "ecg_physionet"
    
    # Validate paths
    if not ecg_path.exists():
        logger.error(f"ECG data path does not exist: {ecg_path}")
        return 1
    
    try:
        # Load metadata
        metadata = load_ptbxl_metadata(str(data_dir))
        
        # Initialize extractor and load model
        extractor = ECGEmbeddingExtractor(model_name=args.model_name)
        extractor.load_model()
        
        # Load raw ECG data
        X_raw = extractor.load_raw_data(metadata, args.sampling_rate, str(ecg_path))
        
        # Preprocess data
        input_tensor = extractor.preprocess_ecg_data(X_raw)
        
        # Extract embeddings
        embeddings = extractor.extract_embeddings(input_tensor, batch_size=args.batch_size)
        
        # Prepare output data
        output_data = {
            'raw': X_raw,
            'emb': embeddings,
            'metadata': metadata
        }
        
        # Save results
        if args.output_file is None:
            output_file = ecg_path / f"ecg_data_and_embeddings_{args.model_name.replace('/', '_')}.npy"
        else:
            output_file = Path(args.root_dir) / "data/ecg_physionet" / Path(args.output_file)
        
        np.save(output_file, output_data)
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Raw data shape: {X_raw.shape}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Output saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())