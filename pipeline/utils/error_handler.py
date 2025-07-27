#!/usr/bin/env python3
"""
MPS-Specific Error Handling and Logging for ASI-Arch

Provides comprehensive error handling for MPS (Apple Silicon) devices,
including automatic fallbacks and detailed error reporting.
"""

import torch
import logging
import traceback
import re
from typing import Optional, Dict, Any, Callable, Union, Tuple
from contextlib import contextmanager
from functools import wraps
from .device_utils import get_optimal_device, move_to_device

logger = logging.getLogger(__name__)


class DeviceError(Exception):
    """Base exception for device-specific errors."""
    pass


class MPSError(DeviceError):
    """Exception for MPS-specific errors."""
    pass


class CUDAError(DeviceError):
    """Exception for CUDA-specific errors."""
    pass


class DeviceErrorHandler:
    """
    Comprehensive error handler for device-specific issues.
    Provides automatic fallbacks and detailed error reporting.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_optimal_device()
        self.device_type = self.device.type
        self.error_counts = {}
        self.fallback_counts = {}
        
        # MPS-specific error patterns
        self.mps_error_patterns = [
            r"MPS.*not implemented",
            r"MPS.*not supported",
            r"MPS.*backend",
            r"operation.*MPS",
            r"MPS.*unsupported",
            r"Metal.*error",
            r"MPSNDArray.*error",
        ]
        
        # CUDA-specific error patterns
        self.cuda_error_patterns = [
            r"CUDA.*out of memory",
            r"CUDA.*error",
            r"cublas.*error",
            r"cuDNN.*error",
            r"GPU.*memory",
        ]
        
        logger.info(f"DeviceErrorHandler initialized for {self.device}")
    
    def is_mps_error(self, error: Exception) -> bool:
        """Check if error is MPS-specific."""
        error_str = str(error)
        return any(re.search(pattern, error_str, re.IGNORECASE) for pattern in self.mps_error_patterns)
    
    def is_cuda_error(self, error: Exception) -> bool:
        """Check if error is CUDA-specific."""
        error_str = str(error)
        return any(re.search(pattern, error_str, re.IGNORECASE) for pattern in self.cuda_error_patterns)
    
    def is_device_error(self, error: Exception) -> bool:
        """Check if error is device-specific."""
        return self.is_mps_error(error) or self.is_cuda_error(error)
    
    def classify_error(self, error: Exception) -> str:
        """Classify the type of error."""
        if self.is_mps_error(error):
            return "mps"
        elif self.is_cuda_error(error):
            return "cuda"
        elif "memory" in str(error).lower():
            return "memory"
        elif "not implemented" in str(error).lower():
            return "not_implemented"
        elif "unsupported" in str(error).lower():
            return "unsupported"
        else:
            return "unknown"
    
    def log_error(self, error: Exception, context: str = "", operation: str = ""):
        """Log error with device-specific context."""
        error_type = self.classify_error(error)
        error_key = f"{error_type}_{operation}" if operation else error_type
        
        # Track error counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create detailed error message
        error_msg = f"Device Error on {self.device}:"
        error_msg += f"\n  Type: {error_type}"
        error_msg += f"\n  Operation: {operation}" if operation else ""
        error_msg += f"\n  Context: {context}" if context else ""
        error_msg += f"\n  Error: {str(error)}"
        error_msg += f"\n  Count: {self.error_counts[error_key]}"
        
        if self.is_device_error(error):
            logger.warning(error_msg)
        else:
            logger.error(error_msg)
        
        # Log stack trace for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Stack trace:\n{traceback.format_exc()}")
    
    def log_fallback(self, from_device: str, to_device: str, operation: str):
        """Log fallback operations."""
        fallback_key = f"{from_device}_to_{to_device}_{operation}"
        self.fallback_counts[fallback_key] = self.fallback_counts.get(fallback_key, 0) + 1
        
        logger.info(f"Fallback {self.fallback_counts[fallback_key]}: {operation} "
                   f"({from_device} → {to_device})")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error and fallback statistics."""
        return {
            "device": str(self.device),
            "error_counts": self.error_counts.copy(),
            "fallback_counts": self.fallback_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "total_fallbacks": sum(self.fallback_counts.values()),
        }
    
    @contextmanager
    def error_context(self, operation: str = "", context: str = ""):
        """Context manager for error handling."""
        try:
            yield self
        except Exception as e:
            self.log_error(e, context, operation)
            raise
    
    def with_fallback(self, primary_func: Callable, fallback_func: Callable, 
                     operation: str = "", *args, **kwargs):
        """
        Execute function with automatic fallback on device errors.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            operation: Description of the operation
            *args, **kwargs: Arguments for the functions
        
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            if self.is_device_error(e):
                self.log_error(e, operation=operation)
                self.log_fallback(self.device_type, "cpu", operation)
                return fallback_func(*args, **kwargs)
            else:
                self.log_error(e, operation=operation)
                raise


def device_safe(operation: str = "", fallback_to_cpu: bool = True):
    """
    Decorator for device-safe operations with automatic error handling.
    
    Args:
        operation: Description of the operation
        fallback_to_cpu: Whether to fallback to CPU on device errors
    
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = DeviceErrorHandler()
            
            with handler.error_context(operation=operation or func.__name__):
                if fallback_to_cpu and handler.device_type in ["mps"]:
                    def cpu_fallback(*a, **kw):
                        # Move tensors to CPU, execute, move back
                        cpu_args = []
                        for arg in a:
                            if isinstance(arg, torch.Tensor):
                                cpu_args.append(arg.cpu())
                            else:
                                cpu_args.append(arg)
                        
                        cpu_kwargs = {}
                        for key, value in kw.items():
                            if isinstance(value, torch.Tensor):
                                cpu_kwargs[key] = value.cpu()
                            else:
                                cpu_kwargs[key] = value
                        
                        result = func(*cpu_args, **cpu_kwargs)
                        
                        # Move result back to original device
                        if isinstance(result, torch.Tensor):
                            return result.to(handler.device)
                        elif isinstance(result, (list, tuple)):
                            return type(result)(
                                t.to(handler.device) if isinstance(t, torch.Tensor) else t 
                                for t in result
                            )
                        else:
                            return result
                    
                    return handler.with_fallback(func, cpu_fallback, operation, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


class MPSCompatibilityChecker:
    """
    Check MPS compatibility for various operations and provide recommendations.
    """
    
    def __init__(self):
        self.known_unsupported = {
            "torch.compile": "Use torch.compile=False for MPS",
            "flash_attn": "Use standard attention implementation",
            "mixed_precision": "Disable mixed precision for MPS",
            "some_einsum_ops": "Some einsum operations may need CPU fallback",
            "complex_indexing": "Complex tensor indexing may need CPU fallback",
        }
        
        self.testing_functions = {
            "basic_ops": self._test_basic_ops,
            "attention": self._test_attention,
            "conv": self._test_conv,
            "linear": self._test_linear,
            "activation": self._test_activation,
        }
    
    def check_compatibility(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Run comprehensive MPS compatibility check.
        
        Args:
            device: Device to test (auto-detected if None)
        
        Returns:
            Dictionary with compatibility results
        """
        if device is None:
            device = get_optimal_device()
        
        if device.type != "mps":
            return {"device": str(device), "mps_specific": False, "all_tests_passed": True}
        
        results = {
            "device": str(device),
            "mps_specific": True,
            "tests": {},
            "unsupported_operations": [],
            "recommendations": [],
        }
        
        # Run compatibility tests
        for test_name, test_func in self.testing_functions.items():
            try:
                test_result = test_func(device)
                results["tests"][test_name] = {"passed": True, "details": test_result}
            except Exception as e:
                results["tests"][test_name] = {"passed": False, "error": str(e)}
                results["unsupported_operations"].append(test_name)
        
        # Add recommendations based on results
        for op in results["unsupported_operations"]:
            if op in self.known_unsupported:
                results["recommendations"].append(self.known_unsupported[op])
        
        results["all_tests_passed"] = len(results["unsupported_operations"]) == 0
        
        return results
    
    def _test_basic_ops(self, device: torch.device) -> Dict[str, bool]:
        """Test basic tensor operations."""
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        
        tests = {
            "matmul": lambda: torch.matmul(x, y),
            "add": lambda: x + y,
            "mul": lambda: x * y,
            "softmax": lambda: torch.softmax(x, dim=-1),
            "sum": lambda: x.sum(),
            "transpose": lambda: x.transpose(0, 1),
        }
        
        results = {}
        for name, test in tests.items():
            try:
                test()
                results[name] = True
            except Exception:
                results[name] = False
        
        return results
    
    def _test_attention(self, device: torch.device) -> Dict[str, bool]:
        """Test attention operations."""
        q = torch.randn(2, 8, 10, 64, device=device)
        k = torch.randn(2, 8, 10, 64, device=device)
        v = torch.randn(2, 8, 10, 64, device=device)
        
        tests = {
            "basic_attention": lambda: torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1), v),
            "scaled_dot_product": self._test_scaled_dot_product if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else lambda: True,
        }
        
        results = {}
        for name, test in tests.items():
            try:
                if callable(test):
                    test()
                results[name] = True
            except Exception:
                results[name] = False
        
        return results
    
    def _test_scaled_dot_product(self) -> bool:
        """Test PyTorch's scaled dot product attention."""
        try:
            q = torch.randn(2, 8, 10, 64, device='mps')
            k = torch.randn(2, 8, 10, 64, device='mps')
            v = torch.randn(2, 8, 10, 64, device='mps')
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return True
        except Exception:
            return False
    
    def _test_conv(self, device: torch.device) -> Dict[str, bool]:
        """Test convolution operations."""
        x = torch.randn(1, 3, 32, 32, device=device)
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
        
        tests = {
            "conv2d": lambda: conv(x),
            "conv1d": lambda: torch.nn.functional.conv1d(
                torch.randn(1, 3, 10, device=device),
                torch.randn(16, 3, 3, device=device)
            ),
        }
        
        results = {}
        for name, test in tests.items():
            try:
                test()
                results[name] = True
            except Exception:
                results[name] = False
        
        return results
    
    def _test_linear(self, device: torch.device) -> Dict[str, bool]:
        """Test linear operations."""
        x = torch.randn(10, 20, device=device)
        linear = torch.nn.Linear(20, 30).to(device)
        
        tests = {
            "linear": lambda: linear(x),
            "bmm": lambda: torch.bmm(
                torch.randn(5, 10, 20, device=device),
                torch.randn(5, 20, 15, device=device)
            ),
        }
        
        results = {}
        for name, test in tests.items():
            try:
                test()
                results[name] = True
            except Exception:
                results[name] = False
        
        return results
    
    def _test_activation(self, device: torch.device) -> Dict[str, bool]:
        """Test activation functions."""
        x = torch.randn(10, 20, device=device)
        
        tests = {
            "relu": lambda: torch.relu(x),
            "gelu": lambda: torch.nn.functional.gelu(x),
            "silu": lambda: torch.nn.functional.silu(x),
            "layer_norm": lambda: torch.nn.functional.layer_norm(x, (20,)),
        }
        
        results = {}
        for name, test in tests.items():
            try:
                test()
                results[name] = True
            except Exception:
                results[name] = False
        
        return results


# Global error handler instance
_global_error_handler = None

def get_error_handler(device: Optional[torch.device] = None) -> DeviceErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    
    target_device = device or get_optimal_device()
    
    if (_global_error_handler is None or 
        _global_error_handler.device != target_device):
        _global_error_handler = DeviceErrorHandler(target_device)
    
    return _global_error_handler


# Convenience functions
def log_device_error(error: Exception, context: str = "", operation: str = "", 
                    device: Optional[torch.device] = None):
    """Log device-specific error with context."""
    handler = get_error_handler(device)
    handler.log_error(error, context, operation)


def check_mps_compatibility(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Check MPS compatibility for current device."""
    checker = MPSCompatibilityChecker()
    return checker.check_compatibility(device)


@contextmanager
def device_error_context(operation: str = "", device: Optional[torch.device] = None):
    """Context manager for device error handling."""
    handler = get_error_handler(device)
    with handler.error_context(operation=operation):
        yield handler