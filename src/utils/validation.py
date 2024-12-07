"""
Environment validation utilities for deep learning development.
Provides comprehensive testing of PyTorch and CUDA capabilities.
"""

import torch
import time
import logging
from typing import Dict, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Validates PyTorch environment and GPU capabilities for deep learning tasks."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.device('cuda')
            self.device_properties = torch.cuda.get_device_properties(0)
        else:
            self.device = torch.device('cpu')

    def get_environment_info(self) -> Dict[str, str]:
        """Retrieves basic environment information."""
        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': str(self.cuda_available),
        }

        if self.cuda_available:
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_device': torch.cuda.get_device_name(0),
                'gpu_memory': f"{self.device_properties.total_memory / 1e9:.2f} GB",
                'compute_capability': f"{self.device_properties.major}.{self.device_properties.minor}"
            })

        return info

    def test_matrix_operations(self, sizes: Optional[list] = None) -> Dict[Tuple[int, int], float]:
        """
        Tests matrix multiplication performance for different sizes.
        Critical for transformer attention operations.
        """
        if sizes is None:
            sizes = [(512, 512), (1024, 1024), (2048, 2048)]

        results = {}
        device = self.device

        for size in sizes:
            matrix1 = torch.randn(*size, device=device)
            matrix2 = torch.randn(*size, device=device)

            # Warm-up run
            torch.matmul(matrix1, matrix2)
            if self.cuda_available:
                torch.cuda.synchronize()

            # Timed run
            start_time = time.time()
            _ = torch.matmul(matrix1, matrix2)
            if self.cuda_available:
                torch.cuda.synchronize()

            elapsed = (time.time() - start_time) * 1000  # ms
            results[size] = elapsed

        return results

    def test_attention_mechanism(self,
                               batch_size: int = 32,
                               seq_length: int = 512,
                               hidden_size: int = 768) -> Dict[str, float]:
        """
        Simulates attention mechanism computations to validate performance.
        Uses typical transformer architecture dimensions.
        """
        device = self.device

        # Create attention inputs
        queries = torch.randn(batch_size, seq_length, hidden_size, device=device)
        keys = torch.randn(batch_size, seq_length, hidden_size, device=device)
        values = torch.randn(batch_size, seq_length, hidden_size, device=device)

        start_time = time.time()

        # Simulate attention computation
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(hidden_size)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        _ = torch.matmul(attention_probs, values)

        if self.cuda_available:
            torch.cuda.synchronize()

        elapsed = (time.time() - start_time) * 1000

        memory_used = (torch.cuda.memory_allocated() / 1e9 if self.cuda_available else 0)

        return {
            'attention_time_ms': elapsed,
            'memory_used_gb': memory_used
        }

    def validate_environment(self) -> bool:
        """
        Performs comprehensive environment validation.
        Returns True if all tests pass, False otherwise.
        """
        try:
            logger.info("Starting environment validation...")

            # Log environment information
            env_info = self.get_environment_info()
            for key, value in env_info.items():
                logger.info(f"{key}: {value}")

            # Test matrix operations
            logger.info("\nTesting matrix operations...")
            matrix_results = self.test_matrix_operations()
            for size, time_ms in matrix_results.items():
                logger.info(f"Matrix multiplication {size}: {time_ms:.2f} ms")

            # Test attention mechanism
            logger.info("\nTesting attention mechanism simulation...")
            attention_results = self.test_attention_mechanism()
            for key, value in attention_results.items():
                logger.info(f"{key}: {value:.2f}")

            logger.info("\nValidation completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            return False

if __name__ == "__main__":
    validator = EnvironmentValidator()
    validator.validate_environment()
