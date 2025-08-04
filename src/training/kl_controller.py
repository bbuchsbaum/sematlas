"""
KL Divergence Controller Callback for β-VAE training.

This callback dynamically adjusts the β parameter in β-VAE training to:
1. Monitor KL-to-total-loss ratio
2. Prevent posterior collapse by maintaining adequate KL divergence
3. Automatically adjust β when KL divergence is too low for sustained periods
"""

import torch
import logging
from typing import Optional, Dict, Any, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
    class MockWandB:
        @staticmethod
        def log(data):
            pass
    wandb = MockWandB()

logger = logging.getLogger(__name__)


class KLDivergenceController(Callback):
    """
    Callback to control KL divergence in β-VAE training.
    
    This callback monitors the KL-to-total-loss ratio and dynamically adjusts β
    to prevent posterior collapse while maintaining good reconstruction quality.
    """
    
    def __init__(
        self,
        target_kl_ratio: float = 0.9,  # Target: KL should be 90% of total loss
        beta_increase_factor: float = 1.1,  # Increase β by 10% when needed
        patience: int = 3,  # Wait 3 epochs before adjusting
        min_kl_threshold: float = 0.01,  # Minimum KL to prevent collapse
        initial_beta: float = 1.0,  # Initial β value
        max_beta: float = 10.0,  # Maximum β value to prevent over-regularization
        log_to_wandb: bool = True
    ):
        """
        Initialize KL divergence controller.
        
        Args:
            target_kl_ratio: Target KL-to-total-loss ratio (default: 0.9 = 90%)
            beta_increase_factor: Factor by which to increase β (default: 1.1 = 10% increase)
            patience: Number of epochs to wait before adjusting β
            min_kl_threshold: Minimum KL value to prevent posterior collapse
            initial_beta: Initial β value
            max_beta: Maximum β value
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        super().__init__()
        
        self.target_kl_ratio = target_kl_ratio
        self.beta_increase_factor = beta_increase_factor
        self.patience = patience
        self.min_kl_threshold = min_kl_threshold
        self.initial_beta = initial_beta
        self.max_beta = max_beta
        self.log_to_wandb = log_to_wandb
        
        # Internal state
        self.current_beta = initial_beta
        self.epochs_below_target = 0
        self.epoch_kl_ratios: List[float] = []
        self.epoch_kl_values: List[float] = []
        self.beta_adjustments: List[Dict[str, Any]] = []
        
        logger.info(f"KL Controller initialized:")
        logger.info(f"  Target KL ratio: {target_kl_ratio:.1%}")
        logger.info(f"  Beta increase factor: {beta_increase_factor:.1%}")
        logger.info(f"  Patience: {patience} epochs")
        logger.info(f"  Min KL threshold: {min_kl_threshold:.4f}")
        logger.info(f"  Initial beta: {initial_beta}")
    
    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called at the end of each training epoch."""
        
        # Get current loss components from the lightning module
        if not hasattr(pl_module, 'last_loss_components'):
            logger.warning("Lightning module missing 'last_loss_components' attribute")
            return
        
        loss_components = pl_module.last_loss_components
        
        if loss_components is None:
            logger.warning("No loss components available for KL monitoring")
            return
        
        # Extract loss values
        total_loss = loss_components.get('total_loss', torch.tensor(0.0))
        kl_loss = loss_components.get('kl_loss', torch.tensor(0.0))
        
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        if isinstance(kl_loss, torch.Tensor):
            kl_loss = kl_loss.item()
        
        # Calculate KL ratio
        if total_loss > 1e-8:  # Avoid division by zero
            kl_ratio = kl_loss / total_loss
        else:
            kl_ratio = 0.0
        
        # Store metrics
        self.epoch_kl_ratios.append(kl_ratio)
        self.epoch_kl_values.append(kl_loss)
        
        # Check if KL is below target
        below_target = kl_ratio < self.target_kl_ratio
        below_collapse_threshold = kl_loss < self.min_kl_threshold
        
        if below_target or below_collapse_threshold:
            self.epochs_below_target += 1
        else:
            self.epochs_below_target = 0
        
        # Log current metrics
        current_epoch = trainer.current_epoch
        logger.info(f"Epoch {current_epoch}: KL={kl_loss:.4f}, KL/Total={kl_ratio:.1%}, β={self.current_beta:.3f}")
        
        if below_collapse_threshold:
            logger.warning(f"KL divergence {kl_loss:.4f} below collapse threshold {self.min_kl_threshold:.4f}")
        
        # Check if β adjustment is needed
        if self.epochs_below_target >= self.patience and self.current_beta < self.max_beta:
            old_beta = self.current_beta
            self.current_beta = min(self.current_beta * self.beta_increase_factor, self.max_beta)
            
            adjustment = {
                'epoch': current_epoch,
                'old_beta': old_beta,
                'new_beta': self.current_beta,
                'reason': f"KL ratio {kl_ratio:.1%} below target {self.target_kl_ratio:.1%} for {self.epochs_below_target} epochs",
                'kl_value': kl_loss,
                'kl_ratio': kl_ratio
            }
            
            self.beta_adjustments.append(adjustment)
            
            logger.info(f"β adjusted: {old_beta:.3f} → {self.current_beta:.3f}")
            logger.info(f"Reason: {adjustment['reason']}")
            
            # Update β in the lightning module
            if hasattr(pl_module, 'current_beta'):
                pl_module.current_beta = self.current_beta
            elif hasattr(pl_module, 'beta'):
                pl_module.beta = self.current_beta
            
            # Reset counter after adjustment
            self.epochs_below_target = 0
        
        # Log to Weights & Biases
        if self.log_to_wandb and WANDB_AVAILABLE:
            wandb_metrics = {
                'kl_controller/beta': self.current_beta,
                'kl_controller/kl_loss': kl_loss,
                'kl_controller/kl_ratio': kl_ratio,
                'kl_controller/target_kl_ratio': self.target_kl_ratio,
                'kl_controller/epochs_below_target': self.epochs_below_target,
                'kl_controller/below_collapse_threshold': below_collapse_threshold
            }
            
            wandb.log(wandb_metrics)
    
    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called at the end of validation epoch - log validation KL metrics."""
        
        # Get validation loss components if available
        if hasattr(pl_module, 'last_val_loss_components') and pl_module.last_val_loss_components is not None:
            val_loss_components = pl_module.last_val_loss_components
            
            total_loss = val_loss_components.get('total_loss', torch.tensor(0.0))
            kl_loss = val_loss_components.get('kl_loss', torch.tensor(0.0))
            
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            if isinstance(kl_loss, torch.Tensor):
                kl_loss = kl_loss.item()
            
            if total_loss > 1e-8:
                val_kl_ratio = kl_loss / total_loss
            else:
                val_kl_ratio = 0.0
            
            # Log validation metrics to W&B
            if self.log_to_wandb and WANDB_AVAILABLE:
                wandb_val_metrics = {
                    'kl_controller/val_kl_loss': kl_loss,
                    'kl_controller/val_kl_ratio': val_kl_ratio
                }
                wandb.log(wandb_val_metrics)
    
    def get_current_beta(self) -> float:
        """Get the current β value."""
        return self.current_beta
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Get history of β adjustments."""
        return self.beta_adjustments.copy()
    
    def get_kl_statistics(self) -> Dict[str, Any]:
        """Get KL divergence statistics."""
        if not self.epoch_kl_ratios:
            return {}
        
        import statistics
        
        return {
            'mean_kl_ratio': statistics.mean(self.epoch_kl_ratios),
            'median_kl_ratio': statistics.median(self.epoch_kl_ratios),
            'min_kl_ratio': min(self.epoch_kl_ratios),
            'max_kl_ratio': max(self.epoch_kl_ratios),
            'mean_kl_value': statistics.mean(self.epoch_kl_values),
            'min_kl_value': min(self.epoch_kl_values),
            'max_kl_value': max(self.epoch_kl_values),
            'num_adjustments': len(self.beta_adjustments),
            'final_beta': self.current_beta
        }
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.current_beta = self.initial_beta
        self.epochs_below_target = 0
        self.epoch_kl_ratios.clear()
        self.epoch_kl_values.clear()
        self.beta_adjustments.clear()
        logger.info("KL Controller reset to initial state")


def create_kl_controller(
    target_kl_ratio: float = 0.9,
    beta_increase_factor: float = 1.1,
    patience: int = 3,
    min_kl_threshold: float = 0.01,
    initial_beta: float = 1.0,
    max_beta: float = 10.0,
    log_to_wandb: bool = True
) -> KLDivergenceController:
    """
    Factory function to create a KL divergence controller.
    
    Args:
        target_kl_ratio: Target KL-to-total-loss ratio
        beta_increase_factor: Factor by which to increase β
        patience: Number of epochs to wait before adjusting β
        min_kl_threshold: Minimum KL value to prevent posterior collapse
        initial_beta: Initial β value
        max_beta: Maximum β value
        log_to_wandb: Whether to log to Weights & Biases
        
    Returns:
        Configured KLDivergenceController
    """
    return KLDivergenceController(
        target_kl_ratio=target_kl_ratio,
        beta_increase_factor=beta_increase_factor,
        patience=patience,
        min_kl_threshold=min_kl_threshold,
        initial_beta=initial_beta,
        max_beta=max_beta,
        log_to_wandb=log_to_wandb
    )


if __name__ == "__main__":
    # Test the KL controller
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing KL Divergence Controller...")
    
    # Create controller
    controller = create_kl_controller(
        target_kl_ratio=0.9,
        patience=2,  # Shorter patience for testing
        log_to_wandb=False  # Disable W&B for testing
    )
    
    # Mock lightning module
    class MockModule:
        def __init__(self):
            self.current_beta = 1.0
            self.last_loss_components = None
    
    # Mock trainer
    class MockTrainer:
        def __init__(self):
            self.current_epoch = 0
    
    module = MockModule()
    trainer = MockTrainer()
    
    print(f"Initial β: {controller.get_current_beta()}")
    
    # Simulate training epochs with low KL
    for epoch in range(10):
        trainer.current_epoch = epoch
        
        # Simulate loss components with progressively lower KL ratio
        total_loss = 1.0
        kl_loss = 0.5 - epoch * 0.05  # Decreasing KL
        
        module.last_loss_components = {
            'total_loss': torch.tensor(total_loss),
            'kl_loss': torch.tensor(max(kl_loss, 0.001))  # Prevent negative
        }
        
        controller.on_train_epoch_end(trainer, module)
        
        print(f"Epoch {epoch}: β={controller.get_current_beta():.3f}")
    
    # Print statistics
    stats = controller.get_kl_statistics()
    print(f"\nStatistics: {stats}")
    
    adjustments = controller.get_adjustment_history()
    print(f"\nAdjustments made: {len(adjustments)}")
    for adj in adjustments:
        print(f"  Epoch {adj['epoch']}: {adj['old_beta']:.3f} → {adj['new_beta']:.3f}")
    
    print("\n✅ KL Controller test completed!")