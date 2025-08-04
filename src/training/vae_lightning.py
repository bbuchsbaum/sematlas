"""
PyTorch Lightning Module for training ResNet VAE.

Implements the training, validation, and optimization logic for the 3D ResNet VAE
with proper VAE loss (reconstruction + KL divergence) and reparameterization trick.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple
import logging

import pytorch_lightning as pl

# Handle imports for both package and script execution
try:
    from src.models.resnet_vae import ResNetVAE3D, create_resnet_vae
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.models.resnet_vae import ResNetVAE3D, create_resnet_vae


logger = logging.getLogger(__name__)


class VAELightningModule(pl.LightningModule):
    """PyTorch Lightning Module for training ResNet VAE."""
    
    def __init__(
        self,
        latent_dim: int = 32,
        groups: int = 8,
        learning_rate: float = 1e-4,
        beta_vae: float = 1.0,
        beta_schedule: str = "constant",  # "constant", "linear", "cyclical"
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
        reconstruction_loss: str = "mse",  # "mse", "bce"
        **kwargs
    ):
        """
        Initialize VAE Lightning Module.
        
        Args:
            latent_dim: Dimensionality of latent space
            groups: Number of groups for Group Normalization
            learning_rate: Learning rate for optimizer
            beta_vae: Weight for KL divergence term
            beta_schedule: Schedule for beta annealing
            max_epochs: Maximum number of training epochs
            weight_decay: Weight decay for optimizer
            reconstruction_loss: Type of reconstruction loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the VAE model
        self.vae = create_resnet_vae(latent_dim=latent_dim, groups=groups)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.beta_vae = beta_vae
        self.beta_schedule = beta_schedule
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.reconstruction_loss = reconstruction_loss
        
        # Track metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        logger.info(f"Initialized VAE with latent_dim={latent_dim}, groups={groups}")
        logger.info(f"Beta schedule: {beta_schedule}, initial beta: {beta_vae}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        return self.vae(x)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        return self.vae.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.vae.decode(z)
    
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample from the latent space."""
        if device is None:
            device = next(self.parameters()).device
        return self.vae.sample(num_samples, device)
    
    def get_current_beta(self) -> float:
        """Get current beta value based on schedule."""
        if self.beta_schedule == "constant":
            return self.beta_vae
        elif self.beta_schedule == "linear":
            # Linear annealing from 0 to beta_vae
            progress = min(self.current_epoch / (self.max_epochs * 0.5), 1.0)
            return self.beta_vae * progress
        elif self.beta_schedule == "cyclical":
            # Cyclical annealing with period of 10 epochs
            cycle_progress = (self.current_epoch % 10) / 10.0
            return self.beta_vae * cycle_progress
        else:
            return self.beta_vae
    
    def vae_loss(
        self, 
        reconstruction: torch.Tensor, 
        target: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            reconstruction: Reconstructed output
            target: Target input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = target.size(0)
        
        # Reconstruction loss
        if self.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
        elif self.reconstruction_loss == "bce":
            # For binary cross entropy, sigmoid is applied in the decoder
            recon_loss = F.binary_cross_entropy_with_logits(
                reconstruction, target, reduction='sum'
            )
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")
        
        # Normalize by batch size and number of voxels
        recon_loss = recon_loss / batch_size
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size
        
        # Current beta for annealing
        current_beta = self.get_current_beta()
        
        # Total loss
        total_loss = recon_loss + current_beta * kl_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(current_beta, device=target.device)
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch['volume']  # (B, 1, 91, 109, 91)
        
        # Forward pass
        reconstruction, mu, logvar = self(x)
        
        # Compute loss
        loss, loss_dict = self.vae_loss(reconstruction, x, mu, logvar)
        
        # Log metrics
        self.log('train/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True)
        self.log('train/recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=True)
        self.log('train/kl_loss', loss_dict['kl_loss'], on_step=True, on_epoch=True)
        self.log('train/beta', loss_dict['beta'], on_step=True, on_epoch=True)
        
        # Store for epoch-end logging
        self.training_step_outputs.append({
            'loss': loss_dict['total_loss'].detach(),
            'recon_loss': loss_dict['recon_loss'].detach(),
            'kl_loss': loss_dict['kl_loss'].detach()
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch['volume']
        
        # Forward pass
        reconstruction, mu, logvar = self(x)
        
        # Compute loss
        loss, loss_dict = self.vae_loss(reconstruction, x, mu, logvar)
        
        # Log metrics
        self.log('val/total_loss', loss_dict['total_loss'], on_step=False, on_epoch=True)
        self.log('val/recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True)
        self.log('val/kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True)
        
        # Store for epoch-end logging
        self.validation_step_outputs.append({
            'loss': loss_dict['total_loss'].detach(),
            'recon_loss': loss_dict['recon_loss'].detach(),
            'kl_loss': loss_dict['kl_loss'].detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if not self.training_step_outputs:
            return
            
        # Compute epoch averages
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_recon = torch.stack([x['recon_loss'] for x in self.training_step_outputs]).mean()
        avg_kl = torch.stack([x['kl_loss'] for x in self.training_step_outputs]).mean()
        
        # Log epoch metrics
        self.log('train/epoch_loss', avg_loss)
        self.log('train/epoch_recon', avg_recon)
        self.log('train/epoch_kl', avg_kl)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch.""" 
        if not self.validation_step_outputs:
            return
            
        # Compute epoch averages
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_recon = torch.stack([x['recon_loss'] for x in self.validation_step_outputs]).mean()
        avg_kl = torch.stack([x['kl_loss'] for x in self.validation_step_outputs]).mean()
        
        # Log epoch metrics
        self.log('val/epoch_loss', avg_loss)
        self.log('val/epoch_recon', avg_recon)
        self.log('val/epoch_kl', avg_kl)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Use AdamW with β₂=0.995 for KL stability (as specified in CLAUDE.md)
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.995),  # β₂=0.995 for KL stability
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def get_reconstruction_sample(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get reconstruction sample for visualization."""
        self.eval()
        with torch.no_grad():
            reconstruction, mu, logvar = self(x)
            sample = self.sample(x.size(0))
            
            return {
                'input': x,
                'reconstruction': reconstruction,
                'sample': sample,
                'mu': mu,
                'logvar': logvar
            }


def create_vae_lightning_module(
    latent_dim: int = 32,
    groups: int = 8,
    learning_rate: float = 1e-4,
    beta_vae: float = 1.0,
    beta_schedule: str = "constant",
    max_epochs: int = 100,
    weight_decay: float = 1e-4,
    reconstruction_loss: str = "mse",
    **kwargs
) -> VAELightningModule:
    """
    Factory function to create a VAELightningModule.
    
    Args:
        latent_dim: Dimensionality of latent space
        groups: Number of groups for Group Normalization
        learning_rate: Learning rate for optimizer
        beta_vae: Weight for KL divergence term
        beta_schedule: Schedule for beta annealing
        max_epochs: Maximum number of training epochs
        weight_decay: Weight decay for optimizer
        reconstruction_loss: Type of reconstruction loss
        **kwargs: Additional arguments
        
    Returns:
        Configured VAELightningModule
    """
    return VAELightningModule(
        latent_dim=latent_dim,
        groups=groups,
        learning_rate=learning_rate,
        beta_vae=beta_vae,
        beta_schedule=beta_schedule,
        max_epochs=max_epochs,
        weight_decay=weight_decay,
        reconstruction_loss=reconstruction_loss,
        **kwargs
    )


if __name__ == "__main__":
    # Test the Lightning Module
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create module
    model = create_vae_lightning_module(
        latent_dim=32,
        groups=8,
        learning_rate=1e-4,
        beta_vae=0.1
    )
    
    print(f"Model created successfully")
    print(f"Latent dim: {model.hparams.latent_dim}")
    print(f"Learning rate: {model.hparams.learning_rate}")
    print(f"Beta VAE: {model.hparams.beta_vae}")
    
    # Test with dummy batch
    batch_size = 2
    dummy_batch = {
        'volume': torch.randn(batch_size, 1, 91, 109, 91),
        'study_id': [f'test_{i}' for i in range(batch_size)],
        'kernel_used': ['6mm'] * batch_size
    }
    
    print(f"\nTesting forward pass...")
    
    # Test training step
    model.train()
    loss = model.training_step(dummy_batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test validation step
    model.eval()
    val_loss = model.validation_step(dummy_batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")
    
    # Test reconstruction sample
    sample_dict = model.get_reconstruction_sample(dummy_batch['volume'])
    print(f"Sample keys: {sample_dict.keys()}")
    print(f"Reconstruction shape: {sample_dict['reconstruction'].shape}")
    print(f"Sample shape: {sample_dict['sample'].shape}")
    
    # Test optimizer configuration
    optimizer_config = model.configure_optimizers()
    print(f"Optimizer type: {type(optimizer_config['optimizer'])}")
    print(f"Scheduler type: {type(optimizer_config['lr_scheduler']['scheduler'])}")
    
    print("\nLightning Module test completed successfully!")