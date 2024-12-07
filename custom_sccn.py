import torch
import torch.nn as nn
from TopoModelX.topomodelx.nn.simplicial.sccn import SCCN
from TopoModelX.topomodelx.nn.simplicial.sccn_layer import SCCNLayer
import torch.nn.functional as F

class GradientSCCNLayer(SCCNLayer):
    """Modified SCCN layer with better gradient flow and handling of missing simplices"""
    def __init__(self, channels, max_rank, aggr_func="sum", update_func="relu", residual=True, is_final_layer=False):
        super().__init__(channels, max_rank, aggr_func, update_func)
        self.residual = residual
        self.is_final_layer = is_final_layer
        
        # Add layer normalization for each rank 
        self.layer_norms = nn.ModuleDict({
            f"rank_{rank}": nn.LayerNorm(channels)
            for rank in range(max_rank + 1)
        })
        
        # Add scaling factors with larger initial values
        self.message_scales = nn.ParameterDict({
            'same_rank': nn.Parameter(torch.ones(1)),
            'low_to_high': nn.Parameter(torch.ones(1)),
            'high_to_low': nn.Parameter(torch.ones(1))
        })
        
        # Add attention-based aggregation
        self.message_attention = nn.ModuleDict({
            f"rank_{rank}": nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, 1)
            ) for rank in range(max_rank + 1)
        })

        # Register gradient hooks
        # self._register_gradient_hooks()

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
            rank_key = f"rank_{rank}"
            
            # Skip if no features for this rank
            if rank_key not in features or features[rank_key] is None:
                out_features[rank_key] = None
                continue
                
            messages = []
            curr_features = features[rank_key]
            
            # Same rank messages with residual
            if rank_key in adjacencies and adjacencies[rank_key] is not None:
                same_rank_msg = self.convs_same_rank[rank_key](
                    curr_features,
                    adjacencies[rank_key]
                ) * self.message_scales['same_rank']
                if self.residual:
                    messages.append(same_rank_msg + curr_features)
                else:
                    messages.append(same_rank_msg)
            
            # Messages from higher rank with residual
            if rank < self.max_rank:
                higher_rank_key = f"rank_{rank+1}"
                if (higher_rank_key in features and 
                    features[higher_rank_key] is not None and
                    higher_rank_key in incidences and 
                    incidences[higher_rank_key] is not None):
                    
                    high_msg = self.convs_high_to_low[rank_key](
                        features[higher_rank_key],
                        incidences[higher_rank_key]
                    ) * self.message_scales['high_to_low']
                    if high_msg.shape == curr_features.shape and self.residual:
                        messages.append(high_msg + curr_features)
                    else:
                        messages.append(high_msg)
            
            # Messages from lower rank with residual
            if rank > 0:
                lower_rank_key = f"rank_{rank-1}"
                curr_incidence_key = rank_key
                if (lower_rank_key in features and 
                    features[lower_rank_key] is not None and
                    curr_incidence_key in incidences and 
                    incidences[curr_incidence_key] is not None):
                    
                    low_msg = self.convs_low_to_high[rank_key](
                        features[lower_rank_key],
                        incidences[curr_incidence_key].transpose(1, 0)
                    ) * self.message_scales['low_to_high']
                    if low_msg.shape == curr_features.shape and self.residual:
                        messages.append(low_msg + curr_features)
                    else:
                        messages.append(low_msg)
            
            # If no messages were collected, use current features
            if not messages:
                out_features[rank_key] = curr_features
                continue
                
            # Stack messages and compute attention
            stacked_msgs = torch.stack(messages)
            attn = self.message_attention[rank_key](stacked_msgs)
            attn = F.softmax(attn, dim=0)
            
            out = (stacked_msgs * attn).sum(dim=0)
            if self.training and not self.is_final_layer:
                out = self.layer_norms[rank_key](out)
            
            out_features[rank_key] = out
            
        return out_features

class GradientSCCN(SCCN):
    """SCCN with improved gradient flow and handling of missing simplices"""
    def __init__(self, channels, max_rank, n_layers=2, update_func="sigmoid", residual=False):
        super().__init__(channels, max_rank, n_layers, update_func)
        
        self.max_rank = max_rank
        # Replace layers with gradient-friendly versions
        self.layers = nn.ModuleList([
            GradientSCCNLayer(
                channels=channels,
                max_rank=max_rank,
                update_func=update_func,
                is_final_layer=(i == n_layers-1)  # True for last layer
            )
            for i in range(n_layers)
        ])

    def forward(self, features, incidences, adjacencies):
        """Forward pass with handling for missing simplices"""
        # Process through each layer
        for layer in self.layers:
            features = layer(features, incidences, adjacencies)
        return features

class JumpingKnowledgeSCCN(GradientSCCN):
    def __init__(self, channels, max_rank, n_layers=2, update_func="relu", residual=False):
        super().__init__(channels, max_rank, n_layers, update_func, residual)
        
        # Add an LSTM to process outputs from all layers
        self.jk_lstm = nn.LSTM(input_size=channels, hidden_size=channels, num_layers=2, batch_first=True)

    def forward(self, features, incidences, adjacencies):
        # Collect outputs from each layer for each rank
        layer_outputs = {f"rank_{rank}": [] for rank in range(self.max_rank + 1)}
        
        # Process through each layer
        curr_features = features
        for layer in self.layers:
            curr_features = layer(curr_features, incidences, adjacencies)
            # Store outputs for each rank if they exist
            for rank in range(self.max_rank + 1):
                rank_key = f"rank_{rank}"
                if rank_key in curr_features and curr_features[rank_key] is not None:
                    layer_outputs[rank_key].append(curr_features[rank_key])
        
        # Process each rank's outputs through the LSTM if they exist
        combined_features = {}
        for rank in range(self.max_rank + 1):
            rank_key = f"rank_{rank}"
            rank_outputs = layer_outputs[rank_key]
            
            # Skip if no outputs for this rank
            if not rank_outputs:
                combined_features[rank_key] = None
                continue
                
            # Stack outputs from all layers for this rank
            stacked_features = torch.stack(rank_outputs, dim=1)  # [batch_size, n_layers, channels]
            
            # Pass through the LSTM
            lstm_out, _ = self.jk_lstm(stacked_features)
            
            # Use the output from the last LSTM cell
            combined_features[rank_key] = lstm_out[:, -1, :]  # [batch_size, channels]
        
        return combined_features