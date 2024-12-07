import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from torch.nn.functional import pad
from rave.core import AudioDistanceV1, MultiScaleSTFT

def batch_mean_difference(target: torch.Tensor, value: torch.Tensor, norm: str = 'L1', relative: bool = False):
    """Like mean_difference but preserves batch dimensions"""
    # Handle batched inputs
    diff = target - value
    
    if norm == 'L1':
        # Take absolute mean over all dims except batch
        diff = diff.abs().mean(dim=list(range(1, diff.dim())))
        if relative:
            norm_target = target.abs().mean(dim=list(range(1, target.dim())))
            diff = diff / (norm_target + 1e-7)
        return diff
    elif norm == 'L2':
        # Take squared mean over all dims except batch
        diff = (diff * diff).mean(dim=list(range(1, diff.dim())))
        if relative:
            norm_target = (target * target).mean(dim=list(range(1, target.dim())))
            diff = diff / (norm_target + 1e-7)
        return diff
    else:
        raise ValueError(f'Norm must be either L1 or L2, got {norm}')

class BatchAudioDistance(AudioDistanceV1):
    """Modified AudioDistanceV1 that preserves batch dimensions"""
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for sx, sy in zip(stfts_x, stfts_y):
            logx = torch.log(sx + self.log_epsilon)
            logy = torch.log(sy + self.log_epsilon)

            lin_distance = batch_mean_difference(sx, sy, norm='L2', relative=True)
            log_distance = batch_mean_difference(logx, logy, norm='L1')

            distance = distance + lin_distance + log_distance

        return {'spectral_distance': distance}

def compute_distances(audio_dir: Path, save_path: Path, batch_size=32):
    """Compute pairwise STFT distances between all audio files"""
    # Set device
    device = torch.device('cpu')  # Stick with CPU for now due to STFT MPS issues
    print(f"Using device: {device}")
    
    # Load all audio files
    audio_files = list(audio_dir.glob('*.wav'))
    n_files = len(audio_files)
    print(f"Found {n_files} audio files")
    
    # Initialize distance matrix and audio distance
    distances = torch.zeros((n_files, n_files))
    distance_fn = BatchAudioDistance(
        lambda: MultiScaleSTFT(scales=[2048, 1024, 512, 256, 128], magnitude=True, sample_rate=16000),
        1e-7
    ).to(device)
    
    # Create mapping of filenames to indices
    file_to_idx = {str(f): i for i, f in enumerate(audio_files)}
    
    # Load and preprocess all audio files
    with torch.no_grad():
        all_audio = []
        max_len = 0
        print("Loading audio files...")
        for f in tqdm(audio_files):
            wav, _ = torchaudio.load(f)
            wav = wav.unsqueeze(0)  # Add batch dimension
            max_len = max(max_len, wav.shape[2])
            all_audio.append(wav)
        
        # Pad all sequences to same length
        all_audio = [pad(w, (0, max_len - w.shape[2])) for w in all_audio]
        all_audio = torch.cat(all_audio, dim=0).to(device)
        print(f"All audio tensor shape: {all_audio.shape}")
        
        # Generate indices for all combinations
        rows, cols = torch.triu_indices(n_files, n_files, offset=1)
        n_pairs = len(rows)
        print(f"Total number of pairs: {n_pairs}")
        
        # Process in batches
        total_batches = (n_pairs + batch_size - 1) // batch_size
        print("\nComputing distances...")
        for b in range(0, n_pairs, batch_size):
            print(f"\rProcessing batch {b//batch_size + 1}/{total_batches} "
                  f"({100*(b/n_pairs):.1f}%)", end="")
            
            # Get indices for this batch
            batch_idx = slice(b, min(b + batch_size, n_pairs))
            batch_rows = rows[batch_idx]
            batch_cols = cols[batch_idx]
            
            # Index directly into all_audio
            batch_i = all_audio[batch_rows]
            batch_j = all_audio[batch_cols]
            
            # Compute distances
            dist = distance_fn(batch_i, batch_j)['spectral_distance']
            
            # Fill distance matrix
            for idx, (i, j) in enumerate(zip(batch_rows, batch_cols)):
                distances[i, j] = dist[idx]
                distances[j, i] = dist[idx]  # Matrix is symmetric
        
        print("\nDistance computation complete!")
    
    print("Sorting distances...")
    # Sort all rows to get complete ordering from nearest to farthest
    sorted_vals, sorted_idx = torch.sort(distances, dim=1)
    
    # Skip the first index (self) for each row
    sorted_vals = sorted_vals[:, 1:]
    sorted_idx = sorted_idx[:, 1:]
    
    # Create neighbors dictionary with complete sorted ordering
    neighbors = {
        str(audio_files[i]): {
            'sorted_neighbors': [str(audio_files[j]) for j in sorted_idx[i].tolist()],
            'sorted_distances': sorted_vals[i].tolist(),
            'index': i
        } for i in range(n_files)
    }
    
    # Add file_to_idx mapping to neighbors dict
    neighbors['__file_to_idx__'] = file_to_idx
    
    # Save results
    print("Saving results...")
    torch.save(distances, save_path / 'distance_matrix.pt')
    with open(save_path / 'neighbors.pkl', 'wb') as f:
        pickle.dump(neighbors, f)
    
    # Save metadata for easy loading
    # metadata = {
    #     'n_files': n_files,
    #     'file_to_idx': file_to_idx,
    # }
    # with open(save_path / 'metadata.json', 'w') as f:
    #     json.dump(metadata, f)
    
    print("Done!")

if __name__ == "__main__":
    audio_dir = Path("./AudioTensors/train")
    save_path = Path("./precomputed")
    save_path.mkdir(exist_ok=True)
    
    compute_distances(audio_dir, save_path) 