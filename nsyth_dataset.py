import torch
import torchaudio
from torch.utils.data import Dataset
import pickle
import random

class NSynthDataset(Dataset):
    def __init__(self, data, root_dir, num_positive_neighbors=10, train=False, num_negative_samples=10):
        """
        data: Dictionary of NSynth metadata.
        root_dir: Directory containing the NSynth audio files.
        num_positive_neighbors: Number of closest neighbors to sample positives from
        initial_negative_offset: How far from the end to start sampling negatives
        """
        self.data = data
        self.root_dir = root_dir
        self.epoch = 0
        self.neighbors = pickle.load(open(f'neighbors.pkl', 'rb'))
        
        # Configuration for positive/negative sampling
        self.num_positive_neighbors = num_positive_neighbors
        self.num_negative_samples = num_negative_samples
        self.initial_negative_offset = len(self.data)
        self.current_negative_offset = self.initial_negative_offset
        
        # Decay rate for negative offset (adjust these parameters as needed)
        self.offset_decay_rate = 0.90  # Will multiply offset by this each epoch
        self.min_negative_offset = 100   # Don't let negatives get closer than this
        self.train = train

    def set_epoch(self, epoch):
        """Update epoch and adjust negative sampling strategy"""
        self.epoch = epoch
        
        # Decay the negative offset
        self.current_negative_offset = max(
            self.min_negative_offset,
            int(self.initial_negative_offset * (self.offset_decay_rate ** epoch))
        )
        
        print(f"Epoch {epoch}: Sampling negatives from {self.current_negative_offset} neighbors from end")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_key = list(self.data.keys())[idx]
        waveform = torch.load(f"{self.root_dir}/{item_key}.pt")
        
        if self.train:
            # Get positive sample
            positive_idx = random.randrange(self.num_positive_neighbors)
            positive_anchor = self.neighbors[item_key]['sorted_neighbors'][positive_idx]
            positive_waveform = torch.load(f"{self.root_dir}/{positive_anchor}.pt")
            
            # Get multiple negative samples
            negative_start = self.current_negative_offset - self.num_negative_samples
            negative_idxs = range(negative_start, self.current_negative_offset)
            negative_anchors = [self.neighbors[item_key]['sorted_neighbors'][i] for i in negative_idxs]
            negative_waveforms = torch.stack([
                torch.load(f"{self.root_dir}/{neg}.pt") for neg in negative_anchors
            ])  # Shape: [num_negatives, channels, time]
            
            # Return all waveforms for contrastive loss
            return torch.cat([
                waveform.unsqueeze(0),
                positive_waveform.unsqueeze(0),
                negative_waveforms
            ])
        
        else:
            return waveform
