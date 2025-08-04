"""
Model inference wrapper for the Generative Brain Atlas.

This module provides a high-level interface for loading trained models and performing
inference, specifically designed for the "Latent Slider" demo and other interactive
applications.
"""

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import warnings

# Add src to path for imports
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

import pytorch_lightning as pl

# Import our modules
try:
    from src.training.vae_lightning import VAELightningModule
    from src.models.resnet_vae import ResNetVAE3D, create_resnet_vae
    from src.data.lightning_datamodule import BrainVolumeDataModule
except ImportError as e:
    raise ImportError(f"Could not import required modules: {e}")


class BrainAtlasInference:
    """
    High-level inference wrapper for the Generative Brain Atlas models.
    
    This class provides an easy-to-use interface for loading trained models
    and performing various inference tasks like latent space traversal,
    brain map generation, and interpolation.
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_dir: str = "inference_cache"):
        """
        Initialize the inference wrapper.
        
        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache generated data
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize attributes
        self.model = None
        self.lightning_model = None
        self.checkpoint_path = checkpoint_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model properties (will be set when model is loaded)
        self.latent_dim = None
        self.input_shape = None
        self.is_trained = False
        
        # Load model if checkpoint provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load with PyTorch Lightning
            self.lightning_model = VAELightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )
            self.model = self.lightning_model.model
            
            # Set model to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            # Extract model properties
            self.latent_dim = self.model.latent_dim
            self.input_shape = (91, 109, 91)  # Standard MNI152 shape
            self.is_trained = True
            self.checkpoint_path = checkpoint_path
            
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Latent dimension: {self.latent_dim}")
            self.logger.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def load_untrained_model(self, latent_dim: int = 128, groups: int = 8) -> None:
        """
        Load an untrained model for testing/demo purposes.
        
        Args:
            latent_dim: Latent space dimensionality
            groups: Group normalization groups
        """
        self.logger.info("Loading untrained model for demo/testing")
        
        self.model = create_resnet_vae(latent_dim=latent_dim, groups=groups)
        self.model.eval()
        self.model.to(self.device)
        
        self.latent_dim = latent_dim
        self.input_shape = (91, 109, 91)
        self.is_trained = False
        
        self.logger.info(f"Untrained model loaded (latent_dim={latent_dim})")
    
    @torch.no_grad()
    def encode(self, brain_volume: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a brain volume to latent space.
        
        Args:
            brain_volume: Input brain volume [H, W, D] or [1, 1, H, W, D]
            
        Returns:
            Tuple of (mean, log_variance) in latent space
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        # Convert to tensor and add batch/channel dimensions if needed
        if isinstance(brain_volume, np.ndarray):
            brain_volume = torch.from_numpy(brain_volume).float()
        
        if brain_volume.dim() == 3:
            brain_volume = brain_volume.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif brain_volume.dim() == 4:
            brain_volume = brain_volume.unsqueeze(0)  # Add batch dim
        
        brain_volume = brain_volume.to(self.device)
        
        # Encode
        mu, logvar = self.model.encode(brain_volume)
        return mu, logvar
    
    @torch.no_grad()
    def decode(self, latent_code: Union[np.ndarray, torch.Tensor], 
               metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Decode latent code to brain volume with optional metadata conditioning.
        
        Args:
            latent_code: Latent vector [latent_dim] or [batch_size, latent_dim]
            metadata: Optional metadata dictionary for conditional generation
            
        Returns:
            Reconstructed brain volume [1, 1, H, W, D] or [batch_size, 1, H, W, D]
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        # Convert to tensor and add batch dimension if needed
        if isinstance(latent_code, np.ndarray):
            latent_code = torch.from_numpy(latent_code).float()
        
        if latent_code.dim() == 1:
            latent_code = latent_code.unsqueeze(0)  # Add batch dimension
        
        latent_code = latent_code.to(self.device)
        
        # Check if model supports conditional generation
        if metadata is not None and hasattr(self.model, 'decode') and hasattr(self.model, 'encoder'):
            # Format metadata for model
            formatted_metadata = self._format_metadata(metadata, latent_code.size(0))
            
            # Try conditional decode if model supports it
            try:
                if hasattr(self.model, 'conditional_decode'):
                    reconstruction = self.model.conditional_decode(latent_code, formatted_metadata)
                else:
                    # Fallback: use standard decode
                    reconstruction = self.model.decode(latent_code)
            except Exception as e:
                logger.warning(f"Conditional decode failed, using standard decode: {e}")
                reconstruction = self.model.decode(latent_code)
        else:
            # Standard unconditional decode
            reconstruction = self.model.decode(latent_code)
            
        return reconstruction
    
    def _format_metadata(self, metadata: Dict[str, Any], batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Format metadata dictionary for model consumption.
        
        Args:
            metadata: Raw metadata dictionary
            batch_size: Batch size to expand metadata to
            
        Returns:
            Formatted metadata tensors
        """
        formatted = {}
        
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                # Expand to batch size if needed
                if value.dim() == 1 and value.size(0) == 1:
                    value = value.repeat(batch_size, 1) if value.dim() > 0 else value.repeat(batch_size)
                elif value.dim() == 1 and len(value) > 1:
                    # Assume it's already the right size or one-hot encoded
                    value = value.unsqueeze(0).repeat(batch_size, 1)
                
                formatted[key] = value.to(self.device)
                
            elif isinstance(value, (int, float)):
                # Convert scalar to tensor
                tensor_val = torch.tensor([float(value)], dtype=torch.float32)
                formatted[key] = tensor_val.repeat(batch_size, 1).to(self.device)
                
            elif isinstance(value, (list, np.ndarray)):
                # Convert array to tensor
                tensor_val = torch.tensor(value, dtype=torch.float32)
                if tensor_val.dim() == 1:
                    tensor_val = tensor_val.unsqueeze(0)
                formatted[key] = tensor_val.repeat(batch_size, 1).to(self.device)
        
        return formatted
    
    @torch.no_grad()
    def reconstruct(self, brain_volume: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct a brain volume (encode then decode).
        
        Args:
            brain_volume: Input brain volume
            
        Returns:
            Reconstructed brain volume
        """
        mu, logvar = self.encode(brain_volume)
        # Use mean of latent distribution for reconstruction
        reconstruction = self.decode(mu)
        return reconstruction
    
    @torch.no_grad()
    def sample_latent(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample random latent codes from standard normal distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Random latent codes [num_samples, latent_dim]
        """
        if self.latent_dim is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        return torch.randn(num_samples, self.latent_dim, device=self.device)
    
    @torch.no_grad()
    def generate_random(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate random brain volumes.
        
        Args:
            num_samples: Number of volumes to generate
            
        Returns:
            Generated brain volumes [num_samples, 1, H, W, D]
        """
        latent_codes = self.sample_latent(num_samples)
        return self.decode(latent_codes)
    
    @torch.no_grad()
    def interpolate_latent(self, 
                          start_code: Union[np.ndarray, torch.Tensor],
                          end_code: Union[np.ndarray, torch.Tensor],
                          num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two latent codes.
        
        Args:
            start_code: Starting latent code [latent_dim]
            end_code: Ending latent code [latent_dim]  
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated latent codes [num_steps, latent_dim]
        """
        # Convert to tensors
        if isinstance(start_code, np.ndarray):
            start_code = torch.from_numpy(start_code).float()
        if isinstance(end_code, np.ndarray):
            end_code = torch.from_numpy(end_code).float()
        
        start_code = start_code.to(self.device)
        end_code = end_code.to(self.device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(1)
        
        # Interpolate
        interpolated = start_code.unsqueeze(0) * (1 - alphas) + end_code.unsqueeze(0) * alphas
        return interpolated
    
    @torch.no_grad()
    def interpolate_volumes(self,
                           start_volume: Union[np.ndarray, torch.Tensor],
                           end_volume: Union[np.ndarray, torch.Tensor],
                           num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two brain volumes via latent space.
        
        Args:
            start_volume: Starting brain volume
            end_volume: Ending brain volume
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated brain volumes [num_steps, 1, H, W, D]
        """
        # Encode both volumes
        start_mu, _ = self.encode(start_volume)
        end_mu, _ = self.encode(end_volume)
        
        # Interpolate in latent space
        interpolated_codes = self.interpolate_latent(start_mu.squeeze(0), end_mu.squeeze(0), num_steps)
        
        # Decode interpolated codes
        return self.decode(interpolated_codes)
    
    @torch.no_grad()
    def traverse_latent_dimension(self, 
                                 base_code: Union[np.ndarray, torch.Tensor, None] = None,
                                 dimension: int = 0,
                                 range_vals: Tuple[float, float] = (-3.0, 3.0),
                                 num_steps: int = 11) -> torch.Tensor:
        """
        Traverse a single latent dimension while keeping others fixed.
        
        This is the core function for the "Latent Slider" demo.
        
        Args:
            base_code: Base latent code to modify. If None, uses zeros.
            dimension: Which latent dimension to traverse
            range_vals: (min, max) values for the traversal
            num_steps: Number of steps in the traversal
            
        Returns:
            Generated brain volumes [num_steps, 1, H, W, D]
        """
        if self.latent_dim is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        if dimension >= self.latent_dim:
            raise ValueError(f"Dimension {dimension} >= latent_dim {self.latent_dim}")
        
        # Create base code if not provided
        if base_code is None:
            base_code = torch.zeros(self.latent_dim, device=self.device)
        else:
            if isinstance(base_code, np.ndarray):
                base_code = torch.from_numpy(base_code).float()
            base_code = base_code.to(self.device)
        
        # Create traversal values
        traversal_values = torch.linspace(range_vals[0], range_vals[1], num_steps, device=self.device)
        
        # Create latent codes for traversal
        latent_codes = base_code.unsqueeze(0).repeat(num_steps, 1)
        latent_codes[:, dimension] = traversal_values
        
        # Generate volumes
        return self.decode(latent_codes)
    
    def save_cache(self, data: Any, cache_key: str) -> None:
        """Save data to cache for faster access."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.debug(f"Saved cache: {cache_key}")
    
    def load_cache(self, cache_key: str) -> Any:
        """Load data from cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.debug(f"Loaded cache: {cache_key}")
            return data
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "is_trained": self.is_trained,
            "checkpoint_path": self.checkpoint_path,
            "latent_dim": self.latent_dim,
            "input_shape": self.input_shape,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": type(self.model).__name__
        }


def create_inference_wrapper(checkpoint_path: Optional[str] = None,
                           device: str = 'auto',
                           fallback_to_untrained: bool = True) -> BrainAtlasInference:
    """
    Convenience function to create an inference wrapper.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use for inference
        fallback_to_untrained: If True, load untrained model if checkpoint fails
        
    Returns:
        Configured BrainAtlasInference wrapper
    """
    wrapper = BrainAtlasInference(device=device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            wrapper.load_checkpoint(checkpoint_path)
            return wrapper
        except Exception as e:
            if fallback_to_untrained:
                logging.warning(f"Failed to load checkpoint: {e}. Loading untrained model.")
                wrapper.load_untrained_model()
                return wrapper
            else:
                raise
    
    elif fallback_to_untrained:
        wrapper.load_untrained_model()
        return wrapper
    
    else:
        raise ValueError("No checkpoint provided and fallback disabled")


# Example usage for testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create wrapper with untrained model for demo
    wrapper = create_inference_wrapper(fallback_to_untrained=True)
    
    print("Model Info:")
    print(wrapper.get_model_info())
    
    print("\nTesting latent traversal:")
    # Test latent traversal
    volumes = wrapper.traverse_latent_dimension(
        dimension=0, 
        range_vals=(-2, 2), 
        num_steps=5
    )
    print(f"Generated volumes shape: {volumes.shape}")
    
    print("\nTesting random generation:")
    # Test random generation
    random_volumes = wrapper.generate_random(num_samples=3)
    print(f"Random volumes shape: {random_volumes.shape}")
    
    print("Inference wrapper test completed successfully!")