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
        initial_reg_factor: float = 0.00001,  # Start very small
        reg_growth_rate: float = 5.0,        # Double each epoch
        max_reg_factor: float = 0.001,         # Cap the maximum regularization
        invalid_state_penalty: float = 100.0,
        device: str = 'cpu',
        seed: int = 511990,
        accumulation_steps: int = 4,
        initial_temp: float = 5.0,
        min_temp: float = 0.1,
        temp_decay: float = 0.95  # Temperature decay per epoch
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create parameter groups with different learning rates
        encoder_params = self.model.encoder.parameters()
        decoder_params = self.model.decoder.parameters()
        
        self.optimizer = Adam([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ])
        
        # Initialize dataloaders
        self.g = self._set_seeds(seed)
        self.train_loader = DataLoader(train_dataset, shuffle=True, generator=self.g)
        self.val_loader = DataLoader(val_dataset, shuffle=False, generator=self.g)
        self.test_loader = DataLoader(test_dataset, shuffle=False, generator=self.g)
        
        # Training components
        self.loss_fn = AutoencoderLoss()
        self.initial_reg_factor = initial_reg_factor
        self.reg_growth_rate = reg_growth_rate
        self.max_reg_factor = max_reg_factor
        self.current_reg_factor = initial_reg_factor
        self.loss_fn.binary_entropy_penalty = initial_reg_factor
        self.invalid_state_penalty = invalid_state_penalty
        
        # Metrics and sample tracking
        self.metrics = TrainingMetrics()
        self.best_samples = []
        self.worst_samples = []
        self.num_samples_to_keep = 10
        self.accumulation_steps = accumulation_steps
        self.grad_clip_threshold = 1.0  # Much lower threshold
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay

    def _set_seeds(self, seed: int) -> Generator:
        torch.manual_seed(seed)
        g = Generator(device='cpu')
        g.manual_seed(seed)
        return g

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
        
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load checkpoint and return training state"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        self.best_samples = checkpoint.get('best_samples', [])
        self.worst_samples = checkpoint.get('worst_samples', [])
        
        return {'epoch': checkpoint['epoch']}

    def train(self, hyper_params: Dict) -> None:
        """Train the model, resuming from checkpoint if available"""
        # Check for existing checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        start_epoch = 0
        
        if latest_checkpoint:
            state = self.load_checkpoint(latest_checkpoint)
            start_epoch = state['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        # Continue hyperparameter tuning
        print('Continuing hyperparameter tuning')
        self.tune_hyperparameters(hyper_params, max_epochs=5, start_epoch=start_epoch)

    def _compute_gradient_norms(self, model):
        """Compute gradient norms for encoder and decoder components"""
        norms = {
            'encoder': {
                'band_processors': 0.0,
                'cross_band': 0.0,
                'temporal_reduction': 0.0,
                'mlp': 0.0,
                'embeddings': 0.0,
                'total': 0.0
            },
            'decoder': {
                'sccn': 0.0,
                'query_sequence': 0.0,
                'layer_norms': 0.0,
                'cross_attention': 0.0,
                'projections': 0.0,
                'attention_scale': 0.0,
                'upsample_blocks': [0.0] * 4,
                'residual_projections': 0.0,
                'residual_scales': 0.0,
                'total': 0.0
            }
        }
        
        # Compute encoder norms
        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                if 'band_processors' in name:
                    norms['encoder']['band_processors'] += param_norm ** 2
                elif 'cross_band' in name:
                    norms['encoder']['cross_band'] += param_norm ** 2
                elif 'temporal_reduction' in name:
                    norms['encoder']['temporal_reduction'] += param_norm ** 2
                elif 'mlp' in name:
                    norms['encoder']['mlp'] += param_norm ** 2
                elif any(x in name for x in ['vertex_embeddings', 'edge_projection', 'triangle_projection', 'tetra_projection']):
                    norms['encoder']['embeddings'] += param_norm ** 2
                norms['encoder']['total'] += param_norm ** 2
        
        # Compute decoder norms
        for name, param in model.decoder.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                
                if 'sccn' in name:
                    norms['decoder']['sccn'] += param_norm ** 2
                elif 'query_sequence' in name:
                    norms['decoder']['query_sequence'] += param_norm ** 2
                elif any(x in name for x in ['pre_attention_norm', 'post_attention_norm']):
                    norms['decoder']['layer_norms'] += param_norm ** 2
                elif 'cross_attention' in name:
                    norms['decoder']['cross_attention'] += param_norm ** 2
                elif any(x in name for x in ['query_proj', 'key_proj', 'value_proj']):
                    norms['decoder']['projections'] += param_norm ** 2
                elif 'attention_scale' in name:
                    norms['decoder']['attention_scale'] += param_norm ** 2
                elif 'upsample_blocks' in name:
                    block_idx = int(name.split('.')[1])
                    norms['decoder']['upsample_blocks'][block_idx] += param_norm ** 2
                elif 'residual_projections' in name:
                    norms['decoder']['residual_projections'] += param_norm ** 2
                elif 'residual_scales' in name:
                    norms['decoder']['residual_scales'] += param_norm ** 2
                
                norms['decoder']['total'] += param_norm ** 2
        
        # Take square root of accumulated squares
        for module in norms['encoder']:
            if module != 'upsample_blocks':
                norms['encoder'][module] = norms['encoder'][module] ** 0.5
        for module in norms['decoder']:
            if module != 'upsample_blocks':
                norms['decoder'][module] = norms['decoder'][module] ** 0.5
        for i in range(len(norms['decoder']['upsample_blocks'])):
            norms['decoder']['upsample_blocks'][i] = norms['decoder']['upsample_blocks'][i] ** 0.5
        
        return norms

    def _log_gradient_norms(self, norms):
        """Log gradient norms in a readable format"""
        print("\nGradient Norms:")
        print("Encoder:")
        print(f"  Band Processors: {norms['encoder']['band_processors']:.4f}")
        print(f"  Cross Band: {norms['encoder']['cross_band']:.4f}")
        print(f"  Temporal Reduction: {norms['encoder']['temporal_reduction']:.4f}")
        print(f"  MLP: {norms['encoder']['mlp']:.4f}")
        print(f"  Embeddings: {norms['encoder']['embeddings']:.4f}")
        print(f"  Total: {norms['encoder']['total']:.4f}")
        print("Decoder:")
        print(f"  SCCN: {norms['decoder']['sccn']:.4f}")
        print(f"  Query Sequence: {norms['decoder']['query_sequence']:.4f}")
        print(f"  Layer Norms: {norms['decoder']['layer_norms']:.4f}")
        print(f"  Cross Attention: {norms['decoder']['cross_attention']:.4f}")
        print(f"  Projections: {norms['decoder']['projections']:.4f}")
        print(f"  Attention Scale: {norms['decoder']['attention_scale']:.4f}")
        for i, norm in enumerate(norms['decoder']['upsample_blocks']):
            print(f"  Upsample Block {i}: {norm:.4f}")
        print(f"  Residual Projections: {norms['decoder']['residual_projections']:.4f}")
        print(f"  Residual Scales: {norms['decoder']['residual_scales']:.4f}")
        print(f"  Total: {norms['decoder']['total']:.4f}")

    def _log_sccn_gradients(self):
        """Log gradients for SCCN components after backward pass"""
        print("\nSCCN Gradients after backward:")
        
        for layer_idx, layer in enumerate(self.model.decoder.sccn.layers):
            print(f"\nLayer {layer_idx}:")
            
            # Log gradients for each rank's components
            for rank in range(4):  # 0 to 3
                print(f"\n  Rank {rank}:")
                
                # Conv weights
                for name, param in layer.convs_same_rank[f"rank_{rank}"].named_parameters():
                    if param.grad is not None:
                        print(f"    Same-rank conv {name} grad norm: {param.grad.norm().item():.6f}")
                
                # Message scales (these are shared across layers)
                if layer_idx == 0:  # Only print once since they're shared
                    for scale_type in ['same_rank', 'low_to_high', 'high_to_low']:
                        scale = layer.message_scales[scale_type]
                        if scale.grad is not None:
                            print(f"    {scale_type} scale grad norm: {scale.grad.norm().item():.6f}")
                
                # Attention weights
                for name, param in layer.message_attention[f"rank_{rank}"].named_parameters():
                    if param.grad is not None:
                        print(f"    Attention {name} grad norm: {param.grad.norm().item():.6f}")
                
                # Layer norm parameters
                for name, param in layer.layer_norms[f"rank_{rank}"].named_parameters():
                    if param.grad is not None:
                        print(f"    Layer norm {name} grad norm: {param.grad.norm().item():.6f}")

    def _log_matrix_gradients(self):
        """Log gradients for complex matrices"""
        print("\nComplex Matrix Gradients after backward:")
        
        # Access the complex matrices from the encoder
        complex_matrices = self.model.encoder.complex_matrices

        # Log gradients for adjacency matrices
        for rank, adj_matrix in complex_matrices.adjacencies.items():
            if adj_matrix.requires_grad:
                print(f"Adjacency matrix {rank} requires grad")
                if adj_matrix.grad is not None:
                    print(f"Adjacency matrix {rank} grad norm: {adj_matrix.grad.norm().item()}")
            else:
                print(f"Adjacency matrix {rank} does not require grad")

        # Log gradients for incidence matrices
        for rank, inc_matrix in complex_matrices.incidences.items():
            if inc_matrix.requires_grad:
                print(f"Incidence matrix {rank} requires grad")
                if inc_matrix.grad is not None:
                    print(f"Incidence matrix {rank} grad norm: {inc_matrix.grad.norm().item()}")
            else:
                print(f"Incidence matrix {rank} does not require grad")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        invalid_states = 0
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
            # Zero gradients at start of iteration
            self.optimizer.zero_grad()
            
            # Forward pass
            batch = batch.to(self.device)
            output, loss_component = self.model(batch)
            
            # Compute loss
            if output is None:
                print("Invalid state detected")
                loss = torch.tensor([100.0], requires_grad=True)
                invalid_states += 1
            else:
                loss = self.loss_fn(batch, output, loss_component)
                if i % 10 == 0:
                    self.save_audio_samples(epoch, i, batch, output, loss.item())
                self._update_best_worst_samples(loss.item(), batch, output, loss_component)

            # Backward pass
            loss.backward()
            
            # Log gradients after backward pass
            if i % 10 == 0:
                self._log_sccn_gradients()
                # self._log_matrix_gradients()
            
            # Track metrics
            total_loss += loss.item()
            
            # Optimizer step
            self.optimizer.step()
            
            # Log current status
            if i % 10 == 0:
                norms = self._compute_gradient_norms(self.model)
                self._log_gradient_norms(norms)
        
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            batch = batch.to(self.device)
            output, loss_component = self.model(batch)
            loss = self.loss_fn(batch, output, loss_component).item()
            # total_loss += loss_dic['spectral_distance'].item()
            total_loss += loss
        return total_loss / len(self.val_loader)

    def save_best_worst_samples(self, epoch: int):
        """Save current best and worst samples"""
        for category, samples in [('best', self.best_samples), ('worst', self.worst_samples)]:
            for i, sample in enumerate(samples):
                sample_dir = self.checkpoint_dir / f'samples/epoch_{epoch}/{category}_{i}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Save audio
                torchaudio.save(sample_dir / 'input.wav', sample.input_audio, 16000)
                torchaudio.save(sample_dir / 'output.wav', sample.output_audio, 16000)
                
                # Save metadata and complex
                with open(sample_dir / 'metadata.json', 'w') as f:
                    json.dump({
                        'loss': sample.loss,
                        'complex_data': sample.complex_data
                    }, f)

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Run final evaluation on test set"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            batch = batch.to(self.device)
            output, loss_component = self.model(batch)
            loss = self.loss_fn(batch, output, loss_component).item()
            # loss = loss_dict['spectral_distance'].item()
            total_loss += loss
            
            # Track best/worst samples
            self._update_best_worst_samples(loss, batch, output, self.model.get_complex())
        
        avg_loss = total_loss / len(self.test_loader)
        
        # Save final results
        test_metrics = {
            'avg_loss': avg_loss,
            'best_loss': self.metrics.best_loss,
            'worst_loss': self.metrics.worst_loss
        }
        
        with open(self.checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f)
            
        # Save best/worst samples
        self.save_best_worst_samples('test')
            
        return test_metrics

    def tune_hyperparameters(self, param_grid: Dict, max_epochs: int = 5, start_epoch: int = 0) -> None:
        """
        Tune hyperparameters using grid search
        
        Args:
            param_grid: Dictionary with parameter names and lists of values to try
            max_epochs: Number of epochs to train each parameter combination
            start_epoch: Epoch to resume from
        """
        from itertools import product
        
        # Generate all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        best_params = None
        best_val_loss = float('inf')
        if start_epoch > 0:
            best_val_loss = self.metrics.best_loss
            best_params = self.metrics.best_params
        
        # Try each parameter combination
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            print(f"\nTrying parameters: {params}")
            
            # Update model parameters
            self.optimizer = Adam([
                {'params': self.model.encoder.parameters(), 'lr': params['encoder_lr']},
                {'params': self.model.decoder.parameters(), 'lr': params['decoder_lr']}
            ])
            self.loss_fn.binary_entropy_penalty = params['reg']
            
            # Train for specified epochs
            for epoch in range(start_epoch, max_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params.copy()
                    self.metrics.best_params = best_params
                    self.save_checkpoint(epoch, is_best=True, params=best_params)
            
            # Reset model weights for next parameter combination
            self.model.reset_weights()
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Set model to best parameters
        self.optimizer = Adam([
            {'params': self.model.encoder.parameters(), 'lr': best_params['encoder_lr']},
            {'params': self.model.decoder.parameters(), 'lr': best_params['decoder_lr']}
        ])
        self.loss_fn.extra_penalty_factor = best_params['reg']

    def save_checkpoint(self, epoch: int, is_best: bool = False, params: Dict = None) -> None:
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: If True, this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'best_samples': self.best_samples,
            'worst_samples': self.worst_samples,
            'best_params': params if is_best else self.metrics.best_params
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")

    def _update_best_worst_samples(self, loss: float, input_audio: torch.Tensor, 
                             output_audio: torch.Tensor, complex_data: dict):
        """Update best and worst samples based on loss"""
        sample = AudioSample(loss, input_audio.cpu(), output_audio.cpu(), complex_data)
        
        # Update best samples
        if loss < self.metrics.best_loss or len(self.best_samples) < self.num_samples_to_keep:
            self.best_samples.append(sample)
            self.best_samples.sort(key=lambda x: x.loss)
            self.best_samples = self.best_samples[:self.num_samples_to_keep]
            self.metrics.best_loss = min(self.metrics.best_loss, loss)
        
        # Update worst samples
        if loss > self.metrics.worst_loss or len(self.worst_samples) < self.num_samples_to_keep:
            self.worst_samples.append(sample)
            self.worst_samples.sort(key=lambda x: x.loss, reverse=True)
            self.worst_samples = self.worst_samples[:self.num_samples_to_keep]
            self.metrics.worst_loss = max(self.metrics.worst_loss, loss)

    def save_audio_samples(self, epoch: int, iteration: int, input_audio: torch.Tensor, 
                      output_audio: torch.Tensor, loss: float):
        """Save periodic audio samples during training"""
        sample_dir = self.checkpoint_dir / f'samples/epoch_{epoch}/iteration_{iteration}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert 3D tensors to 2D (batch, channels, samples) -> (channels, samples)
        input_audio = input_audio.squeeze(0).cpu()  # Remove batch dimension
        output_audio = output_audio.squeeze(0).cpu()
        
        # Save audio files
        torchaudio.save(sample_dir / 'input.wav', input_audio, 16000)
        torchaudio.save(sample_dir / 'output.wav', output_audio, 16000)
        
        # Save metadata
        with open(sample_dir / 'metadata.json', 'w') as f:
            json.dump({
                'loss': loss,
                'epoch': epoch,
                'iteration': iteration
            }, f)