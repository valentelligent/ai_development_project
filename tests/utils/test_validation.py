"""
Test suite for environment validation utilities.
Ensures proper functionality of PyTorch and CUDA capabilities testing.
"""

import unittest
import torch
from src.utils.validation import EnvironmentValidator

class TestEnvironmentValidator(unittest.TestCase):
    def setUp(self):
        """Initialize the validator before each test."""
        self.validator = EnvironmentValidator()

    def test_environment_info(self):
        """Test environment information retrieval."""
        info = self.validator.get_environment_info()

        self.assertIn('pytorch_version', info)
        self.assertIn('cuda_available', info)

        if torch.cuda.is_available():
            self.assertIn('cuda_version', info)
            self.assertIn('gpu_device', info)
            self.assertIn('gpu_memory', info)
            self.assertGreater(float(info['gpu_memory'].split()[0]), 0)

    def test_matrix_operations(self):
        """Test matrix multiplication performance with appropriate matrix sizes for modern GPUs."""
        test_sizes = [(2048, 2048)]  # Using larger matrices for measurable performance
        results = self.validator.test_matrix_operations(test_sizes)

        self.assertEqual(len(results), len(test_sizes))
        for size, time_ms in results.items():
            self.assertIsInstance(time_ms, float)
            self.assertGreaterEqual(time_ms, 0)  # Ensuring non-negative execution time

            # Validate reasonable performance bounds for modern GPUs
            if torch.cuda.is_available():
                # Upper bound check: flag if operation takes too long
                self.assertLess(time_ms, 50.0,
                    "Matrix multiplication taking longer than expected for modern GPU")

    def test_attention_mechanism(self):
        """Test attention mechanism simulation."""
        results = self.validator.test_attention_mechanism(
            batch_size=4,  # Smaller batch size for testing
            seq_length=128,
            hidden_size=256
        )

        self.assertIn('attention_time_ms', results)
        self.assertIn('memory_used_gb', results)
        self.assertGreater(results['attention_time_ms'], 0)

        if torch.cuda.is_available():
            self.assertGreater(results['memory_used_gb'], 0)

    def test_environment_validation(self):
        """Test the complete validation process."""
        validation_result = self.validator.validate_environment()
        self.assertTrue(validation_result)

    def test_cuda_capability(self):
        """Test CUDA capability detection."""
        if torch.cuda.is_available():
            info = self.validator.get_environment_info()
            self.assertIn('compute_capability', info)
            major, minor = map(int, info['compute_capability'].split('.'))
            self.assertGreaterEqual(major, 3)  # Modern GPUs have compute capability >= 3.0

    def tearDown(self):
        """Clean up any resources after each test."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    unittest.main()
