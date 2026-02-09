"""
Vision Transformer configuration module.

Centralized hyperparameter management with inline documentation.
Used by data.py, model.py, and training.py for consistent experiment setup.
"""

from dataclasses import dataclass, asdict
import torch

@dataclass
class ViTConfig:
    """Hyperparameters for ViT training on Caltech-101 dataset."""
    
    # === Data pipeline ===
    img_size: int = 225
    """Input image resolution (pixels). Images resized to (img_size, img_size)."""
    
    patch_size: int = 15
    """Patch dimension"""
    
    batch_size: int = 64
    """Samples per batch. Adjust based on GPU memory (≥4 GB VRAM required)."""
    
    num_classes: int = 101
    """Output classes. Matches Caltech-101 dataset."""
    
    # === Model architecture ===
    embed_dim: int = 768
    """Embedding dimension for patch projections (D in original paper)."""
    
    num_heads: int = 4
    """Attention heads per Transformer layer."""
    
    num_layers: int = 12
    """Number of Transformer encoder blocks."""
    
    dropout: float = 0.3
    """Dropout rate for attention and MLP layers (regularization)."""
    
    # === Training ===
    learning_rate: float = 1e-2
    """Base learning rate for optimizer."""

    loss_fn: type = torch.nn.CrossEntropyLoss
    """Loss function class"""
    
    optimizer: type = torch.optim.SGD
    """Optimizer class"""
    
    momentum: float = 0.8
    """Momentum for SGD (ignored if using Adam/AdamW)."""
    
    epochs: int = 50
    """Total training epochs."""
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return asdict(self)