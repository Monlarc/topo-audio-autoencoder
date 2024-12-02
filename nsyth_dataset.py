import torch
import torchaudio
from torch.utils.data import Dataset

class NSynthDataset(Dataset):
    def __init__(self, data, root_dir):
        """
        data: Dictionary of NSynth metadata.
        root_dir: Directory containing the NSynth audio files.
        """
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_key = list(self.data.keys())[idx]
        waveform = torch.load((f"{self.root_dir}/{item_key}.pt"))
        return waveform
