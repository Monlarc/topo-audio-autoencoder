import torch
import torch.nn as nn
from topomodelx.nn.simplicial.sccn import SCCN
from topomodelx.nn.simplicial.sccn_layer import SCCNLayer
import torch.nn.functional as F

class GradientSCCNLayer(SCCNLayer):
    """Modified SCCN layer with better gradient flow"""
    def __init__(self, channels, max_rank, aggr_func="sum", update_func="relu", residual=True):
        super().__init__(channels, max_rank, aggr_func, update_func)
        self.residual = residual
        
        # Add layer normalization for each rank 
        self.layer_norms = nn.ModuleDict({
            f"rank_{rank}": nn.LayerNorm(channels)
            for rank in range(max_rank + 1)
        })
        
        # Add scaling factors with larger initial values
        self.message_scales = nn.ParameterDict({
            'same_rank': nn.Parameter(torch.ones(1) * 5.0),
            'low_to_high': nn.Parameter(torch.ones(1) * 3.0),
            'high_to_low': nn.Parameter(torch.ones(1) * 3.0)
        })
        
        # Add attention-based aggregation
        self.message_attention = nn.ModuleDict({
            f"rank_{rank}": nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, 1)
            ) for rank in range(max_rank + 1)
        })

        # Register gradient hooks
        self._register_gradient_hooks()

    def _register_gradient_hooks(self):
        def scale_gradients(grad, scale=10.0):
            return grad * scale if grad is not None else grad

        # Register hooks for key operations
        for rank in range(self.max_rank + 1):
            # Hook for same-rank messages
            self.convs_same_rank[f"rank_{rank}"].weight.register_hook(
                lambda grad: scale_gradients(grad, 5.0)
            )
            
            # Hook for low-to-high messages
            if rank > 0:
                self.convs_low_to_high[f"rank_{rank}"].weight.register_hook(
                    lambda grad: scale_gradients(grad, 3.0)
                )
            
            # Hook for high-to-low messages
            if rank < self.max_rank:
                self.convs_high_to_low[f"rank_{rank}"].weight.register_hook(
                    lambda grad: scale_gradients(grad, 3.0)
                )

    def forward(self, features, incidences, adjacencies):
        out_features = {}
        
        for rank in range(self.max_rank + 1):
            messages = []
            
            # Same rank messages with residual
            same_rank_msg = self.convs_same_rank[f"rank_{rank}"](
                features[f"rank_{rank}"],
                adjacencies[f"rank_{rank}"]
            ) * self.message_scales['same_rank']
            if self.residual:
                messages.append(same_rank_msg + features[f"rank_{rank}"])
            else:
                messages.append(same_rank_msg)
            
            # Messages from higher rank with residual
            if rank < self.max_rank:
                high_msg = self.convs_high_to_low[f"rank_{rank}"](
                    features[f"rank_{rank+1}"],
                    incidences[f"rank_{rank+1}"]
                ) * self.message_scales['high_to_low']
                if high_msg.shape == features[f"rank_{rank}"].shape and self.residual:
                    messages.append(high_msg + features[f"rank_{rank}"])
                else:
                    messages.append(high_msg)
            
            # Messages from lower rank with residual
            if rank > 0:
                low_msg = self.convs_low_to_high[f"rank_{rank}"](
                    features[f"rank_{rank-1}"],
                    incidences[f"rank_{rank}"].transpose(1, 0)
                ) * self.message_scales['low_to_high']
                if low_msg.shape == features[f"rank_{rank}"].shape and self.residual:
                    messages.append(low_msg + features[f"rank_{rank}"])
                else:
                    messages.append(low_msg)
            
            # Stack messages and compute attention
            stacked_msgs = torch.stack(messages)
            attn = self.message_attention[f"rank_{rank}"](stacked_msgs)
            attn = F.softmax(attn, dim=0)
            
            out = (stacked_msgs * attn).sum(dim=0)
            out = self.layer_norms[f"rank_{rank}"](out)
            
            out_features[f"rank_{rank}"] = out
            
        return out_features

class GradientSCCN(SCCN):
    """SCCN with improved gradient flow"""
    def __init__(self, channels, max_rank, n_layers=2, update_func="sigmoid", residual=False):
        super().__init__(channels, max_rank, n_layers, update_func)
        
        self.max_rank = max_rank
        # Replace layers with gradient-friendly versions
        self.layers = nn.ModuleList(
            GradientSCCNLayer(
                channels=channels,
                max_rank=max_rank,
                update_func=update_func,
            )
            for _ in range(n_layers)
        ) 

class JumpingKnowledgeSCCN(GradientSCCN):
    def __init__(self, channels, max_rank, n_layers=2, update_func="relu", residual=False):
        super().__init__(channels, max_rank, n_layers, update_func, residual)
        
        # Add an LSTM to process outputs from all layers
        self.jk_lstm = nn.LSTM(input_size=channels, hidden_size=channels, num_layers=2, batch_first=True)

    def forward(self, features, incidences, adjacencies):
        # Collect outputs from each layer for each rank
        layer_outputs = {f"rank_{rank}": [] for rank in range(self.max_rank + 1)}
        
        for layer in self.layers:
            features = layer(features, incidences, adjacencies)
            for rank in range(self.max_rank + 1):
                layer_outputs[f"rank_{rank}"].append(features[f'rank_{rank}'])
        
        # Process each rank's outputs through the LSTM
        combined_features = {}
        for rank in range(self.max_rank + 1):
            # Stack outputs from all layers for this rank
            stacked_features = torch.stack(layer_outputs[f"rank_{rank}"], dim=1)  # [batch_size, n_layers, channels]
            
            # Pass through the LSTM
            lstm_out, _ = self.jk_lstm(stacked_features)
            
            # Use the output from the last LSTM cell
            combined_features[f'rank_{rank}'] = lstm_out[:, -1, :]  # [batch_size, channels]
        
        return combined_features