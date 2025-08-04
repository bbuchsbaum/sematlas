#!/usr/bin/env python3
"""
Training script for Generative Brain Atlas project.

This script orchestrates the training of the baseline 3D ResNet VAE with comprehensive
Weights & Biases integration, checkpointing, and early stopping.

Usage:
    python train.py --config configs/baseline_vae.yaml
    python train.py --config configs/baseline_vae.yaml --resume path/to/checkpoint.ckpt
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Import required packages
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Logging will be disabled.")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.training.vae_lightning import VAELightningModule
    from src.data.lightning_datamodule import BrainVolumeDataModule
    from src.models.resnet_vae import create_resnet_vae
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def create_model(config: Dict[str, Any]) -> 'VAELightningModule':
    """Create the VAE Lightning module from config."""
    model_config = config['model']
    training_config = config['training']
    
    # Create the base model
    model = create_resnet_vae(
        latent_dim=model_config['latent_dim'],
        groups=model_config['group_norm_groups']
    )
    
    # Wrap in Lightning module
    lightning_model = VAELightningModule(
        model=model,
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        beta_schedule=training_config.get('beta_schedule', 'linear'),
        beta_max=float(training_config.get('beta_max', 1.0)),
        beta_warmup_epochs=int(training_config.get('beta_warmup_epochs', 10)),
        max_epochs=int(training_config['max_epochs'])
    )
    
    return lightning_model


def create_datamodule(config: Dict[str, Any]) -> 'BrainVolumeDataModule':
    """Create the data module from config."""
    data_config = config['data']
    training_config = config['training']
    
    return BrainVolumeDataModule(
        data_dir=str(Path(data_config['train_split']).parent),
        lmdb_cache=data_config['volumetric_cache_path'],
        batch_size=training_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create PyTorch Lightning callbacks."""
    callbacks = []
    
    # Checkpointing
    if 'checkpointing' in config:
        checkpoint_config = config['checkpointing']
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            mode=checkpoint_config.get('mode', 'min'),
            save_top_k=checkpoint_config.get('save_top_k', 3),
            save_last=checkpoint_config.get('save_last', True),
            dirpath=checkpoint_config.get('dirpath', 'checkpoints'),
            filename=checkpoint_config.get('filename', 'model-{epoch:02d}-{val_loss:.3f}')
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    if 'early_stopping' in config:
        early_stop_config = config['early_stopping']
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_loss'),
            patience=early_stop_config.get('patience', 15),
            mode=early_stop_config.get('mode', 'min'),
            min_delta=early_stop_config.get('min_delta', 0.001)
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(config: Dict[str, Any]) -> Optional['WandbLogger']:
    """Create Weights & Biases logger."""
    if not WANDB_AVAILABLE or 'wandb' not in config:
        return None
    
    wandb_config = config['wandb']
    
    # Create the logger
    logger = WandbLogger(
        project=wandb_config.get('project', 'generative-brain-atlas'),
        entity=wandb_config.get('entity'),
        name=wandb_config.get('name', 'baseline-vae'),
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        save_dir=wandb_config.get('save_dir', 'wandb_logs'),
        log_model=wandb_config.get('log_model', 'all')
    )
    
    # Log the full config to wandb
    logger.experiment.config.update(config)
    
    return logger


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Generative Brain Atlas VAE')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup everything but don\'t actually train')
    parser.add_argument('--test-run', action='store_true',
                       help='Run a quick test with 2 epochs for validation')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Generative Brain Atlas training...")
    logger.info(f"Using config: {args.config}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override config for test runs
    if args.test_run:
        config['training']['max_epochs'] = 2
        config['training']['val_check_interval'] = 1
        config['logging']['log_every_n_steps'] = 1
        logger.info("Test run mode: reduced epochs and increased logging")
    
    # Create model
    try:
        model = create_model(config)
        logger.info(f"Model created: {type(model).__name__}")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return 1
    
    # Create data module
    try:
        datamodule = create_datamodule(config)
        logger.info("Data module created successfully")
    except Exception as e:
        logger.error(f"Failed to create data module: {e}")
        return 1
    
    # Create logger
    wandb_logger = create_logger(config)
    if wandb_logger:
        logger.info("Weights & Biases logger created")
    else:
        logger.warning("Training without W&B logging")
    
    # Create callbacks
    callbacks = create_callbacks(config)
    logger.info(f"Created {len(callbacks)} callbacks")
    
    if args.dry_run:
        logger.info("Dry run mode: exiting before training")
        return 0
    
    # Create trainer
    hardware_config = config.get('hardware', {})
    logging_config = config.get('logging', {})
    
    trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator=hardware_config.get('accelerator', 'auto'),
            devices=hardware_config.get('devices', 1),
            num_nodes=hardware_config.get('num_nodes', 1),
            precision=hardware_config.get('precision', '16-mixed'),
            callbacks=callbacks,
            logger=wandb_logger,
            val_check_interval=logging_config.get('val_check_interval', 1.0),
            log_every_n_steps=logging_config.get('log_every_n_steps', 50),
            gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_train_batches=config.get('limit_train_batches', 1.0),
            limit_val_batches=config.get('limit_val_batches', 1.0)
    )
    
    logger.info("PyTorch Lightning trainer created")
    
    # Start training
    try:
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            trainer.fit(model, datamodule, ckpt_path=args.resume)
        else:
            logger.info("Starting training from scratch")
            trainer.fit(model, datamodule)
            
        logger.info("Training completed successfully!")
        
        # Test the model
        if config.get('data', {}).get('test_split'):
            logger.info("Running final test evaluation...")
            trainer.test(model, datamodule)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("Training script completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)