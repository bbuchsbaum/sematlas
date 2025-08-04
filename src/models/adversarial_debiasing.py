"""
Adversarial de-biasing components with Gradient Reversal Layer (GRL).

This module implements adversarial training to remove systematic bias 
(e.g., publication year trends) from learned representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer implementation.
    
    This layer acts as an identity function during forward pass,
    but reverses (negates) gradients during backward pass with
    a scaling factor lambda.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """Forward pass - identity function."""
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass - reverse gradients with lambda scaling."""
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) module.
    
    Implements Domain-Adversarial Neural Networks (DANN) style gradient reversal
    for adversarial de-biasing of learned representations.
    """
    
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal with current lambda value."""
        return GradientReversalFunction.apply(x, self.lambda_val)
    
    def set_lambda(self, lambda_val: float):
        """Update lambda value for gradient scaling."""
        self.lambda_val = lambda_val
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_val


class AdversarialMLP(nn.Module):
    """
    Adversarial MLP head for bias prediction.
    
    This network attempts to predict biasing factors (e.g., publication year)
    from learned representations. The GRL ensures that the main encoder
    learns representations that fool this adversarial classifier.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Adversarial classifier network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights with small values for stable training
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adversarial network.
        
        Args:
            x: Input features of shape (B, input_dim)
            
        Returns:
            Predictions of shape (B, output_dim)
        """
        return self.network(x)


class AdversarialDebiasing(nn.Module):
    """
    Complete adversarial de-biasing module with GRL and adversarial head.
    
    This module combines gradient reversal with adversarial prediction
    to remove systematic bias from learned representations.
    """
    
    def __init__(self, feature_dim: int, bias_type: str = 'year', 
                 lambda_val: float = 1.0):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.bias_type = bias_type
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_val)
        
        # Adversarial head based on bias type
        if bias_type == 'year':
            # Regression for publication year (continuous)
            self.adversarial_head = AdversarialMLP(feature_dim, 64, 1)
            self.loss_fn = nn.MSELoss()
        elif bias_type == 'year_binned':
            # Classification for binned years (e.g., decade bins)
            num_bins = 10  # Decades from 1990s to 2020s
            self.adversarial_head = AdversarialMLP(feature_dim, 64, num_bins)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown bias type: {bias_type}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRL and adversarial head.
        
        Args:
            features: Input features of shape (B, feature_dim)
            
        Returns:
            Adversarial predictions
        """
        # Apply gradient reversal
        reversed_features = self.grl(features)
        
        # Make adversarial predictions
        predictions = self.adversarial_head(reversed_features)
        
        return predictions
    
    def compute_adversarial_loss(self, features: torch.Tensor, 
                                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial loss for de-biasing.
        
        Args:
            features: Input features of shape (B, feature_dim)
            targets: Target bias labels (years or binned years)
            
        Returns:
            Adversarial loss scalar
        """
        predictions = self.forward(features)
        
        if self.bias_type == 'year':
            # Regression loss for continuous year prediction
            loss = self.loss_fn(predictions.squeeze(), targets.float())
        elif self.bias_type == 'year_binned':
            # Classification loss for binned years
            loss = self.loss_fn(predictions, targets.long())
        
        return loss
    
    def set_lambda(self, lambda_val: float):
        """Update lambda value for gradient reversal strength."""
        self.grl.set_lambda(lambda_val)
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.grl.lambda_val


class AdversarialLambdaScheduler:
    """
    Lambda scheduler for adversarial training.
    
    Implements various scheduling strategies for the gradient reversal
    lambda parameter during training.
    """
    
    def __init__(self, schedule_type: str = 'linear_ramp', 
                 start_epoch: int = 20, end_epoch: int = 80,
                 lambda_max: float = 1.0):
        self.schedule_type = schedule_type
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.lambda_max = lambda_max
        
    def get_lambda(self, epoch: int) -> float:
        """
        Get lambda value for current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Lambda value for gradient reversal
        """
        if self.schedule_type == 'constant':
            return self.lambda_max
        
        elif self.schedule_type == 'linear_ramp':
            if epoch < self.start_epoch:
                return 0.0
            elif epoch > self.end_epoch:
                return self.lambda_max
            else:
                # Linear ramp from 0 to lambda_max
                progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
                return progress * self.lambda_max
        
        elif self.schedule_type == 'exponential_ramp':
            if epoch < self.start_epoch:
                return 0.0
            elif epoch > self.end_epoch:
                return self.lambda_max
            else:
                # Exponential ramp (faster growth later)
                progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
                return self.lambda_max * (progress ** 2)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def create_year_bins(years: torch.Tensor, start_year: int = 1990, bin_size: int = 5) -> torch.Tensor:
    """
    Create binned year labels for classification.
    
    Args:
        years: Tensor of publication years
        start_year: Starting year for binning
        bin_size: Size of each bin in years
        
    Returns:
        Binned year labels as integers
    """
    # Clamp years to reasonable range
    years_clamped = torch.clamp(years, start_year, start_year + bin_size * 10)
    
    # Create bins
    bins = ((years_clamped - start_year) / bin_size).long()
    bins = torch.clamp(bins, 0, 9)  # 10 bins total (0-9)
    
    return bins


def create_mock_year_data(batch_size: int, bias_type: str = 'year') -> torch.Tensor:
    """
    Create mock publication year data for testing.
    
    Args:
        batch_size: Number of samples
        bias_type: Type of bias ('year' or 'year_binned')
        
    Returns:
        Mock year data
    """
    # Generate random years between 1995 and 2023
    years = torch.randint(1995, 2024, (batch_size,))
    
    if bias_type == 'year':
        return years.float()
    elif bias_type == 'year_binned':
        return create_year_bins(years)
    else:
        raise ValueError(f"Unknown bias type: {bias_type}")


if __name__ == "__main__":
    # Test adversarial de-biasing components
    print("Testing Adversarial De-biasing Components...")
    
    batch_size = 4
    feature_dim = 256
    
    # Test Gradient Reversal Layer
    print("\n1. Testing Gradient Reversal Layer...")
    grl = GradientReversalLayer(lambda_val=0.5)
    features = torch.randn(batch_size, feature_dim, requires_grad=True)
    
    # Forward pass
    reversed_features = grl(features)
    assert reversed_features.shape == features.shape, "GRL should preserve shape"
    print(f"✅ GRL forward pass: {features.shape} -> {reversed_features.shape}")
    
    # Test lambda update
    grl.set_lambda(1.0)
    assert grl.get_lambda() == 1.0, "Lambda update failed"
    print("✅ GRL lambda update works")
    
    # Test Adversarial MLP
    print("\n2. Testing Adversarial MLP...")
    adv_mlp = AdversarialMLP(feature_dim, 64, 1)
    with torch.no_grad():
        predictions = adv_mlp(features)
        assert predictions.shape == (batch_size, 1), f"Wrong prediction shape: {predictions.shape}"
    print(f"✅ Adversarial MLP: {feature_dim} -> {predictions.shape}")
    
    # Test complete adversarial de-biasing
    print("\n3. Testing Complete Adversarial De-biasing...")
    debiaser = AdversarialDebiasing(feature_dim, bias_type='year')
    
    # Create mock year targets
    year_targets = create_mock_year_data(batch_size, 'year')
    
    # Test forward pass
    with torch.no_grad():
        adv_predictions = debiaser(features)
        assert adv_predictions.shape == (batch_size, 1), "Wrong adversarial prediction shape"
        
        # Test loss computation
        adv_loss = debiaser.compute_adversarial_loss(features, year_targets)
        assert adv_loss.dim() == 0, "Loss should be scalar"
        assert adv_loss >= 0, "Loss should be non-negative"
        
    print(f"✅ Adversarial predictions: {adv_predictions.shape}")
    print(f"✅ Adversarial loss: {adv_loss.item():.4f}")
    
    # Test lambda scheduler
    print("\n4. Testing Lambda Scheduler...")
    scheduler = AdversarialLambdaScheduler('linear_ramp', 20, 80, 1.0)
    
    test_epochs = [0, 10, 20, 50, 80, 100]
    for epoch in test_epochs:
        lambda_val = scheduler.get_lambda(epoch)
        print(f"Epoch {epoch:3d}: λ = {lambda_val:.3f}")
    
    print("✅ Lambda scheduler working")
    
    # Test year binning
    print("\n5. Testing Year Binning...")
    years = torch.tensor([1995, 2000, 2005, 2010, 2015, 2020])
    binned = create_year_bins(years, start_year=1990, bin_size=5)
    print(f"Years: {years.tolist()}")
    print(f"Bins:  {binned.tolist()}")
    print("✅ Year binning works")
    
    print("\n🎉 All adversarial de-biasing components working!")