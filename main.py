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

@dataclass
class DataConfig:
    train_samples: int = 1000
    val_ratio: float = 0.2
    base_path: str = "/Users/carlking/CS6170_Project/NSynth"

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
        for key in nsynth_data.keys():
            self.preprocess_audio(key, audio_path, save_path)
            
        return NSynthDataset(nsynth_data, save_path)

    def get_datasets(self) -> Tuple[NSynthDataset, NSynthDataset, NSynthDataset]:
        """Get train, validation, and test datasets"""
        train_data = self.process_split("train", self.config.train_samples)
        val_samples = int(self.config.train_samples * self.config.val_ratio)
        val_data = self.process_split("valid", val_samples)
        test_data = self.process_split("test", val_samples)
        
        return train_data, val_data, test_data

def main():
    # Configuration
    data_config = DataConfig()
    hyper_params = {
        'encoder_lr': [0.001, 0.0001],
        'decoder_lr': [0.00001, 0.000001],
        'reg': [0, 1e-7]
    }
    
    # Initialize model and data processor
    model = AudioAutoencoder(num_vertices=20, num_bands=16, sccn_hidden_dim=64)
    data_processor = DataProcessor(data_config)
    
    # Get datasets
    train_data, val_data, test_data = data_processor.get_datasets()

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        checkpoint_dir='./checkpoints'
    )
    trainer.train(hyper_params)
    
    print(f"Model parameters: {model.num_params():,}")

if __name__ == "__main__":
    main()