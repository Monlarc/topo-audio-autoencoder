import torch
import torchaudio
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from nsyth_dataset import NSynthDataset
from trainer import Trainer
from audio2complex import AudioAutoencoder
from dataclasses import dataclass
import random
import numpy as np
import pickle
from IPython.display import Audio, display
from tqdm import tqdm
from precompute_distances import compute_distances
import soundfile
@dataclass
class DataConfig:
    train_samples: int = 1024
    val_ratio: float = 0.2
    base_path: str = "/Users/carlking/CS6170_Project/NSynth"
    precomputed_path: str = "./precomputed"

class DataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.rng = random.Random(511990)  # For reproducibility
        
    def preprocess_audio(self, key: str, root_dir: str, save_dir: Path) -> None:
        """Preprocess and save a single audio file"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = Path(root_dir) / f"{key}.wav"
        raw_waveform, _ = torchaudio.load(str(audio_path))
        
        # Save the waveform tensor
        torch.save(raw_waveform, save_dir / f"{key}.pt")
        
        # Copy the wav file for distance computation
        import shutil
        wav_save_dir = save_dir / "wav"
        wav_save_dir.mkdir(exist_ok=True)
        shutil.copy(audio_path, wav_save_dir / f"{key}.wav")

    def process_split(self, split: str, max_samples: Optional[int] = None) -> NSynthDataset:
        """Process a data split (train/valid/test) with random sampling"""
        nsynth_path = Path(self.config.base_path) / f"nsynth-{split}"
        json_path = nsynth_path / "examples.json"
        audio_path = nsynth_path / "audio"
        save_path = Path("./AudioTensors") / split
        
        with open(json_path, 'r') as file:
            nsynth_data = json.load(file)
            
        # Random sampling if max_samples specified
        if max_samples and max_samples < len(nsynth_data):
            keys = list(nsynth_data.keys())
            selected_keys = self.rng.sample(keys, max_samples)
            nsynth_data = {k: nsynth_data[k] for k in selected_keys}
            
        # Process each audio file
        for key in tqdm(nsynth_data.keys(), desc=f"Processing {split} split"):
            self.preprocess_audio(key, audio_path, save_path)
        
        # Compute distances for this split if it's training data
        if split == "train":
            precomputed_path = Path(self.config.precomputed_path)
            precomputed_path.mkdir(exist_ok=True)
            if not (precomputed_path / 'neighbors.pkl').exists():
                print("\nComputing distances for training set...")
                compute_distances(
                    audio_dir=save_path / "wav",
                    save_path=precomputed_path
                )
                print("Distance computation complete!")
        
        return NSynthDataset(nsynth_data, save_path, train=split == "train")

    def get_datasets(self) -> Tuple[NSynthDataset, NSynthDataset, NSynthDataset]:
        """Get train, validation, and test datasets"""
        train_data = self.process_split("train", self.config.train_samples)
        val_samples = int(self.config.train_samples * self.config.val_ratio)
        val_data = self.process_split("valid", val_samples)
        test_data = self.process_split("test", val_samples)
        
        return train_data, val_data, test_data
def explore_neighbors(data_config: DataConfig, num_neighbors: int = 3):
    """Save original audio and its nearest/farthest neighbors for exploration"""
    try:
        # Load precomputed neighbors
        with open(Path(data_config.precomputed_path) / 'neighbors.pkl', 'rb') as f:
            neighbors = pickle.load(f)
        
        # Load distance matrix
        distances = torch.load(Path(data_config.precomputed_path) / 'distance_matrix.pt')
        
        # Create directory for neighbor audio files
        neighbor_dir = Path(data_config.precomputed_path) / 'neighbor_samples'
        neighbor_dir.mkdir(parents=True, exist_ok=True)
        
        # Pick a random note, excluding the metadata key
        valid_keys = [k for k in neighbors.keys() if k != '__file_to_idx__']
        if not valid_keys:
            raise ValueError("No valid audio files found in neighbors dictionary")
        
        sample_key = random.choice(valid_keys)
        print(f"\nSaving neighbors for: {sample_key}")
        
        # Create directory for this sample
        sample_name = Path(sample_key).stem
        sample_dir = neighbor_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Save original
        try:
            original_audio, sr = torchaudio.load(sample_key)
            if sr != 16000:
                original_audio = torchaudio.transforms.Resample(sr, 16000)(original_audio)
            torchaudio.save(sample_dir / 'original.wav', original_audio, 16000)
        except Exception as e:
            print(f"Error loading original audio {sample_key}: {e}")
            return
        
        # Save nearest neighbors
        nearest_dir = sample_dir / 'nearest'
        nearest_dir.mkdir(exist_ok=True)
        
        for i, neighbor in enumerate(neighbors[sample_key]['sorted_neighbors'][:num_neighbors]):
            try:
                dist = neighbors[sample_key]['sorted_distances'][i]
                audio, sr = torchaudio.load(neighbor)
                if sr != 16000:
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
                torchaudio.save(
                    nearest_dir / f'neighbor_{i+1}_dist_{dist:.4f}.wav',
                    audio,
                    16000
                )
            except Exception as e:
                print(f"Error processing nearest neighbor {neighbor}: {e}")
                continue
        
        # Save farthest neighbors
        farthest_dir = sample_dir / 'farthest'
        farthest_dir.mkdir(exist_ok=True)
        
        for i, neighbor in enumerate(neighbors[sample_key]['sorted_neighbors'][-num_neighbors:]):
            try:
                # Get distance from the end of the sorted distances
                dist = neighbors[sample_key]['sorted_distances'][-(i+1)]
                audio, sr = torchaudio.load(neighbor)
                if sr != 16000:
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
                torchaudio.save(
                    farthest_dir / f'neighbor_{i+1}_dist_{dist:.4f}.wav',
                    audio,
                    16000
                )
            except Exception as e:
                print(f"Error processing farthest neighbor {neighbor}: {e}")
                continue
        
        print(f"\nSaved audio files to {sample_dir}")
        print("\nDirectory structure:")
        print(f"{sample_dir}/")
        print("├── original.wav")
        print("├── nearest/")
        for i in range(num_neighbors):
            print(f"│   ├── neighbor_{i+1}_dist_X.XXX.wav")
        print("└── farthest/")
        for i in range(num_neighbors):
            print(f"    ├── neighbor_{i+1}_dist_X.XXX.wav")
            
    except Exception as e:
        print(f"An error occurred: {e}")
# def explore_neighbors(data_config: DataConfig, num_neighbors: int = 3):
#     """Save original audio and its nearest/farthest neighbors for exploration"""
#     # Load precomputed neighbors
#     with open(Path(data_config.precomputed_path) / 'neighbors.pkl', 'rb') as f:
#         neighbors = pickle.load(f)

#     # Load distance matrix
#     distances = torch.load(Path(data_config.precomputed_path) / 'distance_matrix.pt')

#     # Create directory for neighbor audio files
#     neighbor_dir = Path(data_config.precomputed_path) / 'neighbor_samples'
#     neighbor_dir.mkdir(parents=True, exist_ok=True)
    
#     # Pick a random note
#     sample_key = random.choice(list(neighbors.keys()))
#     print(f"\nSaving neighbors for: {sample_key}")
    
#     # Create directory for this sample
#     sample_name = Path(sample_key).stem
#     sample_dir = neighbor_dir / sample_name
#     sample_dir.mkdir(exist_ok=True)
    
#     # Save original
#     original_audio, _ = torchaudio.load(sample_key)
#     torchaudio.save(sample_dir / 'original.wav', original_audio, 16000)
    
#     # Save nearest neighbors
#     nearest_dir = sample_dir / 'nearest'
#     nearest_dir.mkdir(exist_ok=True)
#     for i, neighbor in enumerate(neighbors[sample_key]['sorted_neighbors'][:num_neighbors]):
#         dist = neighbors[sample_key]['sorted_distances'][i]
#         audio, _ = torchaudio.load(neighbor)
#         torchaudio.save(
#             nearest_dir / f'neighbor_{i+1}_dist_{dist:.4f}.wav', 
#             audio, 
#             16000
#         )
    
#     # Save farthest neighbors
#     farthest_dir = sample_dir / 'farthest'
#     farthest_dir.mkdir(exist_ok=True)
#     for i, neighbor in enumerate(neighbors[sample_key]['sorted_neighbors'][-num_neighbors:]):
#         dist = neighbors[sample_key]['sorted_distances'][i]
#         audio, _ = torchaudio.load(neighbor)
#         torchaudio.save(
#             farthest_dir / f'neighbor_{i+1}_dist_{dist:.4f}.wav',
#             audio,
#             16000
#         )
    
#     print(f"\nSaved audio files to {sample_dir}")
#     print("\nDirectory structure:")
#     print(f"{sample_dir}/")
#     print("├── original.wav")
#     print("├── nearest/")
#     print("│   ├── neighbor_1_dist_X.XXX.wav")
#     print("│   ├── neighbor_2_dist_X.XXX.wav")
#     print("│   └── neighbor_3_dist_X.XXX.wav")
#     print("└── farthest/")
#     print("    ├── neighbor_1_dist_X.XXX.wav")
#     print("    ├── neighbor_2_dist_X.XXX.wav")
#     print("    └── neighbor_3_dist_X.XXX.wav")

def setup_checkpoint_dir(base_path: str = './checkpoints') -> Path:
    """Setup checkpoint directory, handling existing directories"""
    checkpoint_dir = Path(base_path)
    old_checkpoint_dir = Path(f"{base_path}_old")
    
    # If old checkpoint directory exists, remove it
    if old_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(old_checkpoint_dir)
    
    # If checkpoint directory exists, rename it to old
    if checkpoint_dir.exists():
        checkpoint_dir.rename(old_checkpoint_dir)
    
    # Create fresh checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def main():
    # Configuration
    data_config = DataConfig()
    hyper_params = {
        'encoder_lr': [0.001, 0.0005],
        'decoder_lr': [0.0001, 0.00005],
        'complexity_penalty': [0.05, 0.1]
    }
    
    
    # Setup checkpoint directory
    checkpoint_dir = setup_checkpoint_dir()
    
    # Initialize model and data processor
    model = AudioAutoencoder(
        num_vertices=20, 
        num_bands=16, 
        sccn_hidden_dim=64,
        min_active_vertices=8,
        max_active_vertices=20
    )
    data_processor = DataProcessor(data_config)
    
    # Get datasets
    train_data, val_data, test_data = data_processor.get_datasets()

    # Optionally explore neighbors
    if input("Explore neighbors? (y/n): ").lower() == 'y':
        explore_neighbors(data_config)
        if input("Continue with training? (y/n): ").lower() != 'y':
            return

    # Create trainer with precomputed distances
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        checkpoint_dir=checkpoint_dir,
        gradient_clip_val=10.0,
        accumulate_grad_batches=4,
        warmup_epochs=5,
        initial_reg_factor=0.00001,
        reg_growth_rate=5.0,
        max_reg_factor=0.001,
        precomputed_path=data_config.precomputed_path  # Add this
    )
    trainer.train(hyper_params)
    
    print(f"Model parameters: {model.num_params():,}")

if __name__ == "__main__":
    main()