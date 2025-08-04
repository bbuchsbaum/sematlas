"""
Inference module for the Generative Brain Atlas.

This module provides high-level interfaces for loading trained models and performing
various inference tasks including latent space traversal, brain map generation,
and interactive visualization.
"""

from .model_wrapper import BrainAtlasInference, create_inference_wrapper

__all__ = ['BrainAtlasInference', 'create_inference_wrapper']