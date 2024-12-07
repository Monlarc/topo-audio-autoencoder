import torch
from torch.utils.data import DataLoader
from torch import Generator
from torch.optim import Adam
from tqdm import tqdm 
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torchaudio
from loss import AutoencoderLoss
torch.autograd.set_detect_anomaly(True)

@dataclass
class TrainingMetrics:
    train_losses: List[float] = None
    val_losses: List[float] = None
    iteration_losses: List[Tuple[int, int, float]] = None  # (epoch, iteration, loss)
    best_loss: float = float('inf')
    worst_loss: float = float('-inf')
    best_epoch: int = 0
    best_params: Dict = None  # Store the best hyperparameters
    
    def __post_init__(self):
        self.train_losses = []
        self.val_losses = []
        self.iteration_losses = []
    
    def save(self, save_dir: Path):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'iteration_losses': self.iteration_losses,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'best_params': self.best_params
        }
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

@dataclass
class AudioSample:
    loss: float
    input_audio: torch.Tensor
    output_audio: torch.Tensor
    complex_data: dict

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        checkpoint_dir: str,
        encoder_lr: float = 1e-3,
        decoder_lr: float = 1e-4,
        initial_reg_factor: float = 0.00001,
        reg_growth_rate: float = 5.0,
        max_reg_factor: float = 0.001,
        invalid_state_penalty: float = 100.0,
        device: str = 'cpu',
        seed: int = 511990,
        initial_temp: float = 5.0,
        min_temp: float = 0.1,
        temp_decay: float = 0.95,
        gradient_clip_val: float = 10.0,
        warmup_epochs: int = 5,
        accumulate_grad_batches: int = 4,
        precomputed_path: Path = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed_path = precomputed_path
        
        # Create parameter groups with different learning rates
        encoder_params = self.model.encoder.parameters()
        decoder_params = self.model.decoder.parameters()
        
        self.optimizer = Adam([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ])
        
        # Initialize dataloaders
        self.g = self._set_seeds(seed)
        self.train_dataset = train_dataset

        self.train_loader = DataLoader(train_dataset, shuffle=True, generator=self.g)
        self.val_loader = DataLoader(val_dataset, shuffle=False, generator=self.g)
        self.test_loader = DataLoader(test_dataset, shuffle=False, generator=self.g)
        
        # Training components
        self.loss_fn = AutoencoderLoss(
            binary_entropy_penalty=initial_reg_factor,
            min_entropy_penalty=0.01,
            complexity_penalty=0.1
        )
        self.initial_reg_factor = initial_reg_factor
        self.reg_growth_rate = reg_growth_rate
        self.max_reg_factor = max_reg_factor
        self.current_reg_factor = initial_reg_factor
        self.invalid_state_penalty = invalid_state_penalty
        
        # Training control parameters
        self.accumulation_steps = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.warmup_epochs = warmup_epochs
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        
        # Metrics and sample tracking
        self.metrics = TrainingMetrics()
        
    def train(self, hyper_params=None):
        """Train the model with optional hyperparameter tuning"""
        if hyper_params:
            print("Starting hyperparameter tuning...")
            self.tune_hyperparameters(hyper_params)
            # Load best parameters
            self.load_best_parameters()
            
        print("Starting full training...")
        max_epochs = 100
        patience = 20
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            self.train_dataset.set_epoch(epoch)
            self.metrics.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.metrics.val_losses.append(val_loss)
            
            # Save metrics
            self.metrics.save(self.checkpoint_dir)
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
            # Regular checkpoint and audio samples
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}')
                
    def save_audio_samples(self, epoch: int, iteration: int, input_audio: torch.Tensor, output_audio: torch.Tensor):
        """Save audio samples from current model state"""
        self.model.eval()
        sample_dir = self.checkpoint_dir / f'samples/epoch_{epoch}_iter_{iteration}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio files
        torchaudio.save(sample_dir / f'input_{iteration}.wav', input_audio.squeeze(0), 16000)
        torchaudio.save(sample_dir / f'output_{iteration}.wav', output_audio.squeeze(0), 16000)
        
        # Save metadata
        with open(sample_dir / f'metadata_{iteration}.json', 'w') as f:
            json.dump({
                'complex_data': {
                    'num_vertices': len(self.model.encoder.active_simplices['vertices']),
                    'num_edges': len(self.model.encoder.active_simplices['edges']),
                    'num_triangles': len(self.model.encoder.active_simplices['triangles']),
                    'num_tetra': len(self.model.encoder.active_simplices['tetra'])
                }
            }, f, indent=2)
        
        self.model.train()
        
    def tune_hyperparameters(self, hyper_params):
        """Tune hyperparameters using grid search"""
        best_val_loss = float('inf')
        best_params = None
        tuning_epochs = 5
        
        # Try each parameter combination
        from itertools import product
        param_values = [hyper_params['encoder_lr'], hyper_params['decoder_lr'], hyper_params['complexity_penalty']]
        
        for encoder_lr, decoder_lr, complexity_penalty in product(*param_values):
            # Create directory name from hyperparameters
            param_dir = self.checkpoint_dir / f'e{encoder_lr}_d{decoder_lr}_c{complexity_penalty}'
            param_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nTrying parameters: encoder_lr={encoder_lr}, decoder_lr={decoder_lr}, complexity_penalty={complexity_penalty}")
            print(f"Checkpoint directory: {param_dir}")
            
            # Try to load latest checkpoint
            latest_checkpoint = self._get_latest_checkpoint(param_dir)
            if latest_checkpoint:
                print(f"Loading checkpoint: {latest_checkpoint}")
                self.load_checkpoint(latest_checkpoint, param_dir)
                start_epoch = int(latest_checkpoint.stem.split('_')[1])  # Extract epoch from filename
            else:
                print("No checkpoint found, starting fresh")
                start_epoch = 0
                # Update parameters
                self.optimizer.param_groups[0]['lr'] = encoder_lr
                self.optimizer.param_groups[1]['lr'] = decoder_lr
                self.loss_fn.complexity_penalty = complexity_penalty
                # Reset model weights
                self.model.reset_weights()
            
            # Train for remaining epochs
            for epoch in range(start_epoch, tuning_epochs):
                train_loss = self.train_epoch(epoch, param_dir)
                val_loss = self.validate()
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(f'epoch_{epoch}', param_dir)
                
                # Check if these parameters are better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        'encoder_lr': encoder_lr,
                        'decoder_lr': decoder_lr,
                        'complexity_penalty': complexity_penalty
                    }
                    self.save_checkpoint('best', param_dir)
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save best parameters
        self.metrics.best_params = best_params
        self.metrics.save(self.checkpoint_dir)
        
    def load_best_parameters(self):
        """Load the best parameters found during tuning"""
        if self.metrics.best_params is None:
            print("No best parameters found, using current parameters")
            return
            
        params = self.metrics.best_params
        self.optimizer.param_groups[0]['lr'] = params['encoder_lr']
        self.optimizer.param_groups[1]['lr'] = params['decoder_lr']
        self.loss_fn.complexity_penalty = params['complexity_penalty']
        
        # Load best model weights
        self.load_checkpoint('best_tuning')
        
    def train_epoch(self, epoch, param_dir: Optional[Path] = None):
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # Update temperature
        self.model.encoder.sampler.current_temp = max(
            self.min_temp,
            self.current_temp * (self.temp_decay ** epoch)
        )
        
        # for iteration, x in tqdm(enumerate(self.train_loader), desc=f"Training Epoch {epoch}", total=len(self.train_loader)):
        for iteration, x in enumerate(self.train_loader):
            x = x.to(self.device)
            
            # Forward pass
            output, diversity_loss = self.model(x)
            
            if output is None:  # Invalid state
                loss = torch.tensor(self.invalid_state_penalty, device=self.device)
            else:
                loss = self.loss_fn(output, x, diversity_loss)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (iteration + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            
            if iteration % 10 == 0:
                # Log metrics
                print(f"\nIteration {iteration}, Loss: {loss.item() * self.accumulation_steps:.4f}")
                grad_norms = self._compute_gradient_norms(self.model)
                self._log_gradient_norms(grad_norms)
                self.save_audio_samples(epoch, iteration, x, output)

            if iteration % 100 == 0:
                self.save_checkpoint(f'epoch_{epoch}_iter_{iteration}', param_dir)

            
            total_loss += loss.item() * self.accumulation_steps
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        return avg_loss
        
    def _compute_gradient_norms(self, model):
        """Compute L2 gradient norms for encoder and decoder components"""
        # Initialize accumulators for squared norms
        norms = {
            'encoder': {
                'band_processors': 0.0,
                'skip_weight': 0.0,
                'cross_band': 0.0,
                'temporal_reduction': 0.0,
                'to_simplices': 0.0,
                "biases": 0.0,
                'embeddings': 0.0,
                'total': 0.0
            },
            'decoder': {
                'sccn': 0.0,
                'layer_norms': 0.0,
                'cross_attention': 0.0,
                'projections': 0.0,
                'attention_scale': 0.0,
                'upsample_blocks': [0.0] * 4,
                'total': 0.0
            }
        }
        
        # Accumulate squared norms
        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                if 'band_processors' in name:
                    norms['encoder']['band_processors'] += param_norm ** 2
                elif 'cross_band' in name:
                    norms['encoder']['cross_band'] += param_norm ** 2
                elif 'temporal_reduction' in name:
                    norms['encoder']['temporal_reduction'] += param_norm ** 2
                elif 'to_simplices' in name:
                    norms['encoder']['to_simplices'] += param_norm ** 2
                elif 'bias' in name:
                    norms['encoder']['biases'] += param_norm ** 2
                elif 'skip_weight' in name:
                    norms['encoder']['skip_weight'] += param_norm ** 2
                elif any(x in name for x in ['vertex_embeddings', 'edge_projection', 'triangle_projection', 'tetra_projection']):
                    norms['encoder']['embeddings'] += param_norm ** 2
                norms['encoder']['total'] += param_norm ** 2
        
        for name, param in model.decoder.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                if 'sccn' in name:
                    norms['decoder']['sccn'] += param_norm ** 2
                elif any(x in name for x in ['pre_attention_norm', 'post_attention_norm']):
                    norms['decoder']['layer_norms'] += param_norm ** 2
                elif 'cross_attention' in name:
                    norms['decoder']['cross_attention'] += param_norm ** 2
                elif any(x in name for x in ['query_proj', 'key_proj', 'value_proj']):
                    norms['decoder']['projections'] += param_norm ** 2
                elif 'upsample_blocks' in name:
                    block_idx = int(name.split('.')[1])
                    norms['decoder']['upsample_blocks'][block_idx] += param_norm ** 2
                norms['decoder']['total'] += param_norm ** 2
        
        # Take square root of accumulated squared norms
        for module in ['encoder', 'decoder']:
            for key in norms[module]:
                if isinstance(norms[module][key], list):
                    norms[module][key] = [x ** 0.5 for x in norms[module][key]]
                else:
                    norms[module][key] = norms[module][key] ** 0.5
        
        return norms
        
    def _log_gradient_norms(self, norms):
        """Log gradient norms in a readable format"""
        print("\nGradient Norms:")
        print("Encoder:")
        for k, v in norms['encoder'].items():
            if k != 'upsample_blocks':
                print(f"  {k}: {v:.4f}")
        print("Decoder:")
        for k, v in norms['decoder'].items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    print(f"  Upsample Block {i}: {val:.4f}")
            else:
                print(f"  {k}: {v:.4f}")
                
    def validate(self):
        """Run validation and return average loss"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for x in self.val_loader:
                x = x.to(self.device)
                output, entropy_loss = self.model(x)
                
                if output is not None:
                    loss = self.loss_fn(output, x, entropy_loss['binary_entropy'], entropy_loss['diversity'])
                    total_loss += loss.item()
                    batch_count += 1
                
        return total_loss / batch_count if batch_count > 0 else float('inf')
        
    def save_checkpoint(self, name: str, checkpoint_dir: Optional[Path] = None):
        """Save a checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'hyperparameters': {
                'encoder_lr': self.optimizer.param_groups[0]['lr'],
                'decoder_lr': self.optimizer.param_groups[1]['lr'],
                'complexity_penalty': self.loss_fn.complexity_penalty
            }
        }
        torch.save(checkpoint, checkpoint_dir / f'{name}.pt')
        
    def load_checkpoint(self, name: str | Path, checkpoint_dir: Optional[Path] = None):
        """Load a checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        
        if isinstance(name, Path):
            checkpoint_path = name
        else:
            checkpoint_path = checkpoint_dir / f'{name}.pt'
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        
        # Load hyperparameters
        params = checkpoint['hyperparameters']
        self.optimizer.param_groups[0]['lr'] = params['encoder_lr']
        self.optimizer.param_groups[1]['lr'] = params['decoder_lr']
        self.loss_fn.complexity_penalty = params['complexity_penalty']
        
    def _set_seeds(self, seed: int) -> Generator:
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        g = Generator(device='cpu')
        g.manual_seed(seed)
        return g

    def _get_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """Get the path to the latest checkpoint in the directory"""
        checkpoints = list(checkpoint_dir.glob('epoch_*.pt'))
        if not checkpoints:
            return None
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        return checkpoints[-1]