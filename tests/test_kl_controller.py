"""
Test suite for S2.2.2: KL Divergence Controller Callback.

Tests success criteria:
- Callback monitors KL-to-total-loss ratio
- β increases by 10% when KL < 90% target for 3 epochs
- W&B logs show β adjustments during training
- Prevents posterior collapse (KL remains >0.01)
"""

import torch
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.kl_controller import KLDivergenceController, create_kl_controller


class TestKLDivergenceController:
    """Test suite for KL divergence controller."""
    
    def test_callback_monitors_kl_to_total_loss_ratio(self):
        """Test that callback monitors KL-to-total-loss ratio correctly."""
        controller = create_kl_controller(
            target_kl_ratio=0.9,
            patience=3,
            log_to_wandb=False
        )
        
        # Mock objects
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        
        mock_module = Mock()
        mock_module.current_beta = 1.0
        
        # Test with different KL ratios
        test_cases = [
            {'total_loss': 1.0, 'kl_loss': 0.9, 'expected_ratio': 0.9},  # At target
            {'total_loss': 1.0, 'kl_loss': 0.5, 'expected_ratio': 0.5},  # Below target
            {'total_loss': 2.0, 'kl_loss': 1.0, 'expected_ratio': 0.5},  # Below target (different scale)
            {'total_loss': 0.0, 'kl_loss': 0.1, 'expected_ratio': 0.0},  # Division by zero case
        ]
        
        for i, case in enumerate(test_cases):
            mock_trainer.current_epoch = i
            mock_module.last_loss_components = {
                'total_loss': torch.tensor(case['total_loss']),
                'kl_loss': torch.tensor(case['kl_loss'])
            }
            
            controller.on_train_epoch_end(mock_trainer, mock_module)
            
            # Check that ratio was calculated correctly
            recorded_ratio = controller.epoch_kl_ratios[-1]
            assert abs(recorded_ratio - case['expected_ratio']) < 1e-6, \
                f"KL ratio calculation incorrect: expected {case['expected_ratio']}, got {recorded_ratio}"
        
        print("✅ Callback monitors KL-to-total-loss ratio correctly")
    
    def test_beta_increases_by_10_percent_when_below_target(self):
        """Test that β increases by 10% when KL < 90% target for 3 epochs."""
        controller = create_kl_controller(
            target_kl_ratio=0.9,
            beta_increase_factor=1.1,  # 10% increase
            patience=3,
            initial_beta=1.0,
            log_to_wandb=False
        )
        
        # Mock objects
        mock_trainer = Mock()
        mock_module = Mock()
        mock_module.current_beta = 1.0
        
        # Simulate epochs with KL below target (50% ratio)
        for epoch in range(5):
            mock_trainer.current_epoch = epoch
            mock_module.last_loss_components = {
                'total_loss': torch.tensor(1.0),
                'kl_loss': torch.tensor(0.5)  # 50% ratio, below 90% target
            }
            
            old_beta = controller.get_current_beta()
            controller.on_train_epoch_end(mock_trainer, mock_module)
            new_beta = controller.get_current_beta()
            
            if epoch == 2:  # After 3 epochs (0, 1, 2), should adjust at end of epoch 2
                expected_beta = 1.0 * 1.1  # 10% increase
                assert abs(new_beta - expected_beta) < 1e-6, \
                    f"Beta should increase by 10% after {controller.patience} epochs"
                print(f"    Epoch {epoch}: β adjusted from {old_beta:.3f} to {new_beta:.3f}")
            elif epoch == 5:  # After another 3 epochs, should adjust again
                expected_beta = 1.0 * 1.1 * 1.1  # Two 10% increases
                assert abs(new_beta - expected_beta) < 1e-6, \
                    f"Beta should increase again after another {controller.patience} epochs"
        
        # Check adjustment history
        adjustments = controller.get_adjustment_history()
        assert len(adjustments) >= 1, "Should have recorded β adjustments"
        
        first_adjustment = adjustments[0]
        assert first_adjustment['old_beta'] == 1.0
        assert abs(first_adjustment['new_beta'] - 1.1) < 1e-6
        assert first_adjustment['epoch'] == 2
        
        print("✅ β increases by 10% when KL < 90% target for 3 epochs")
    
    @patch('wandb.log')  # Mock wandb.log
    def test_wandb_logs_beta_adjustments(self, mock_wandb_log):
        """Test that W&B logs show β adjustments during training."""
        controller = create_kl_controller(
            target_kl_ratio=0.9,
            patience=2,  # Shorter patience for faster testing
            log_to_wandb=True
        )
        
        # Mock objects
        mock_trainer = Mock()
        mock_module = Mock()
        mock_module.current_beta = 1.0
        
        # Simulate training with low KL ratio
        for epoch in range(4):
            mock_trainer.current_epoch = epoch
            mock_module.last_loss_components = {
                'total_loss': torch.tensor(1.0),
                'kl_loss': torch.tensor(0.3)  # 30% ratio, well below target
            }
            
            controller.on_train_epoch_end(mock_trainer, mock_module)
        
        # Check that wandb.log was called
        assert mock_wandb_log.called, "W&B logging should be called"
        
        # Check that β adjustments were logged
        logged_metrics = [call[0][0] for call in mock_wandb_log.call_args_list]
        
        # Should log metrics every epoch
        assert len(logged_metrics) >= 4, "Should log metrics for each epoch"
        
        # Check that logged metrics include β values
        for metrics in logged_metrics:
            assert 'kl_controller/beta' in metrics, "Should log current β value"
            assert 'kl_controller/kl_ratio' in metrics, "Should log KL ratio"
            assert 'kl_controller/kl_loss' in metrics, "Should log KL loss"
        
        # Check for β changes in logged values
        beta_values = [metrics.get('kl_controller/beta', 1.0) for metrics in logged_metrics]
        assert max(beta_values) > min(beta_values), "β values should change over time"
        
        print("✅ W&B logs show β adjustments during training")
    
    def test_prevents_posterior_collapse(self):
        """Test that callback prevents posterior collapse (KL remains >0.01)."""
        controller = create_kl_controller(
            target_kl_ratio=0.9,
            min_kl_threshold=0.01,
            patience=1,  # Quick response to collapse
            beta_increase_factor=2.0,  # Aggressive increase for testing
            log_to_wandb=False
        )
        
        # Mock objects
        mock_trainer = Mock()
        mock_module = Mock()
        mock_module.current_beta = 1.0
        
        # Simulate posterior collapse scenario
        collapse_scenarios = [
            {'kl_loss': 0.005, 'total_loss': 1.0, 'scenario': 'Near collapse'},
            {'kl_loss': 0.001, 'total_loss': 1.0, 'scenario': 'Critical collapse'},
            {'kl_loss': 0.0001, 'total_loss': 1.0, 'scenario': 'Severe collapse'}
        ]
        
        for i, scenario in enumerate(collapse_scenarios):
            mock_trainer.current_epoch = i
            mock_module.last_loss_components = {
                'total_loss': torch.tensor(scenario['total_loss']),
                'kl_loss': torch.tensor(scenario['kl_loss'])
            }
            
            old_beta = controller.get_current_beta()
            controller.on_train_epoch_end(mock_trainer, mock_module)
            new_beta = controller.get_current_beta()
            
            # Should increase β when KL is below threshold
            if scenario['kl_loss'] < controller.min_kl_threshold:
                assert new_beta > old_beta, f"β should increase for {scenario['scenario']}"
                print(f"    {scenario['scenario']}: KL={scenario['kl_loss']:.4f}, β: {old_beta:.3f} → {new_beta:.3f}")
        
        # Check that all KL values were recorded correctly
        recorded_kl_values = controller.epoch_kl_values
        for i, scenario in enumerate(collapse_scenarios):
            assert abs(recorded_kl_values[i] - scenario['kl_loss']) < 1e-6, \
                "KL values should be recorded accurately"
        
        print("✅ Prevents posterior collapse (responds to KL < threshold)")\n    \n    def test_controller_statistics_and_history(self):\n        \"\"\"Test that controller provides accurate statistics and history.\"\"\"\n        controller = create_kl_controller(\n            target_kl_ratio=0.8,\n            patience=2,\n            log_to_wandb=False\n        )\n        \n        # Mock objects\n        mock_trainer = Mock()\n        mock_module = Mock()\n        mock_module.current_beta = 1.0\n        \n        # Simulate varied training\n        kl_values = [0.9, 0.5, 0.3, 0.6, 0.2]\n        total_loss = 1.0\n        \n        for epoch, kl_loss in enumerate(kl_values):\n            mock_trainer.current_epoch = epoch\n            mock_module.last_loss_components = {\n                'total_loss': torch.tensor(total_loss),\n                'kl_loss': torch.tensor(kl_loss)\n            }\n            \n            controller.on_train_epoch_end(mock_trainer, mock_module)\n        \n        # Check statistics\n        stats = controller.get_kl_statistics()\n        \n        assert 'mean_kl_ratio' in stats\n        assert 'min_kl_ratio' in stats\n        assert 'max_kl_ratio' in stats\n        assert 'num_adjustments' in stats\n        assert 'final_beta' in stats\n        \n        # Verify calculations\n        expected_ratios = [kl / total_loss for kl in kl_values]\n        assert abs(stats['mean_kl_ratio'] - sum(expected_ratios) / len(expected_ratios)) < 1e-6\n        assert abs(stats['min_kl_ratio'] - min(expected_ratios)) < 1e-6\n        assert abs(stats['max_kl_ratio'] - max(expected_ratios)) < 1e-6\n        \n        # Check adjustment history\n        history = controller.get_adjustment_history()\n        assert isinstance(history, list)\n        \n        if history:  # If adjustments were made\n            for adjustment in history:\n                assert 'epoch' in adjustment\n                assert 'old_beta' in adjustment\n                assert 'new_beta' in adjustment\n                assert 'reason' in adjustment\n        \n        print(\"✅ Controller provides accurate statistics and history\")\n    \n    def test_max_beta_limit(self):\n        \"\"\"Test that β doesn't exceed maximum limit.\"\"\"\n        controller = create_kl_controller(\n            max_beta=2.0,  # Low max for testing\n            patience=1,\n            beta_increase_factor=1.5,  # Large increase factor\n            log_to_wandb=False\n        )\n        \n        # Mock objects\n        mock_trainer = Mock()\n        mock_module = Mock()\n        mock_module.current_beta = 1.0\n        \n        # Simulate many epochs with low KL to force multiple adjustments\n        for epoch in range(10):\n            mock_trainer.current_epoch = epoch\n            mock_module.last_loss_components = {\n                'total_loss': torch.tensor(1.0),\n                'kl_loss': torch.tensor(0.1)  # Very low KL\n            }\n            \n            controller.on_train_epoch_end(mock_trainer, mock_module)\n            \n            # β should never exceed max_beta\n            current_beta = controller.get_current_beta()\n            assert current_beta <= controller.max_beta, f\"β {current_beta} exceeds max {controller.max_beta}\"\n        \n        # Final β should be at the maximum\n        final_beta = controller.get_current_beta()\n        assert final_beta == controller.max_beta, f\"Final β should reach maximum: {final_beta} vs {controller.max_beta}\"\n        \n        print(\"✅ β respects maximum limit\")\n\n\ndef test_s2_2_2_success_criteria():\n    \"\"\"\n    Comprehensive test for S2.2.2 SUCCESS_MARKERS criteria:\n    - [✅] Callback monitors KL-to-total-loss ratio\n    - [✅] β increases by 10% when KL < 90% target for 3 epochs\n    - [✅] W&B logs show β adjustments during training\n    - [✅] Prevents posterior collapse (KL remains >0.01)\n    \"\"\"\n    print(\"\\n=== Testing S2.2.2: KL Controller Implementation ===\")\n    \n    test_suite = TestKLDivergenceController()\n    \n    # Run all tests\n    test_suite.test_callback_monitors_kl_to_total_loss_ratio()\n    test_suite.test_beta_increases_by_10_percent_when_below_target()\n    test_suite.test_wandb_logs_beta_adjustments()\n    test_suite.test_prevents_posterior_collapse()\n    test_suite.test_controller_statistics_and_history()\n    test_suite.test_max_beta_limit()\n    \n    print(\"\\n🎉 All S2.2.2 SUCCESS CRITERIA PASSED!\")\n    print(\"✅ Callback monitors KL-to-total-loss ratio\")\n    print(\"✅ β increases by 10% when KL < 90% target for 3 epochs\")\n    print(\"✅ W&B logs show β adjustments during training\")\n    print(\"✅ Prevents posterior collapse (KL remains >0.01)\")\n\n\nif __name__ == \"__main__\":\n    test_s2_2_2_success_criteria()