"""
Neural network architectures for FashionMNIST classification:
"""

import torch
import torch.nn as nn
from typing import Literal
from config import ModelConfig


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Name of activation function ("relu", "tanh", "gelu", "silu")
    """
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


class MLP(nn.Module):
    """
    Multi-layer perceptron (fully connected feedforward network).
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        layers = []
        dims = [cfg.input_dim] + list(cfg.hidden_dims) + [cfg.output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # No activation/norm/dropout after final layer
            if i < len(dims) - 2:
                if cfg.use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(get_activation(cfg.activation))
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier/Glorot for tanh-like, He/Kaiming for ReLU-like)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights appropriately for the activation function."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.cfg.activation in ["relu", "gelu", "silu"]:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualMLP(nn.Module):
    """MLP with residual connections.
    
    Physics motivation: Residual connections help with gradient flow,
    analogous to how small perturbations propagate in linear systems.
    The network learns f(x) = x + g(x), where g is a small correction.
    
    This is useful for learning small deviations from a baseline or
    for very deep networks where gradients might vanish.
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Input projection if input_dim != first hidden dim
        if cfg.input_dim != cfg.hidden_dims[0]:
            self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dims[0])
        else:
            self.input_proj = nn.Identity()
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i, dim in enumerate(cfg.hidden_dims):
            next_dim = cfg.hidden_dims[i + 1] if i + 1 < len(cfg.hidden_dims) else dim
            self.blocks.append(ResidualBlock(dim, next_dim, cfg))
        
        # Output projection
        self.output_proj = nn.Linear(cfg.hidden_dims[-1], cfg.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Single residual block: y = x + MLP(x)"""
    
    def __init__(self, in_dim: int, out_dim: int, cfg: ModelConfig):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim) if cfg.use_layer_norm else nn.Identity()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.activation = get_activation(cfg.activation)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


# ============================================================================
# Model factory
# ============================================================================

def create_model(cfg: ModelConfig, architecture: str = "mlp") -> nn.Module:
    """Create a model from configuration.
    
    Args:
        cfg: Model configuration
        architecture: "mlp" or "residual_mlp"
    
    Returns:
        PyTorch model
    """
    architectures = {
        "mlp": MLP,
        "residual_mlp": ResidualMLP,
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(architectures.keys())}")
    
    model = architectures[architecture](cfg)
    print(f"Created {architecture} with {model.count_parameters():,} parameters")
    return model