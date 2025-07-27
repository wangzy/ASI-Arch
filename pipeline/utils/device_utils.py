#!/usr/bin/env python3
"""
Device Detection and Management Utilities for ASI-Arch

Provides automatic device detection and optimization for CUDA, MPS (Apple Silicon), and CPU.
Handles device-specific optimizations and fallbacks for different hardware configurations.
"""

import torch
import platform
import logging
from typing import Dict, Any, Tuple, Optional
import warnings

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """
    Automatically detect the best available device for training and inference.
    
    Priority order:
    1. CUDA (if available and functional)
    2. MPS (Apple Silicon - if available and functional) 
    3. CPU (fallback)
    
    Returns:
        torch.device: The optimal device for computation
    """
    # Check CUDA availability first (highest performance)
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality with a simple operation
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor + 1
            logger.info(f"CUDA detected: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"CUDA available but not functional: {e}")
    
    # Check MPS availability (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test MPS functionality
            test_tensor = torch.tensor([1.0]).to('mps')
            test_result = test_tensor + 1
            logger.info("Apple Silicon MPS detected and functional")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS available but not functional: {e}")
    
    # Fallback to CPU
    logger.info("Using CPU device (CUDA and MPS unavailable)")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for logging and optimization.
    
    Returns:
        Dict containing device information and capabilities
    """
    device = get_optimal_device()
    info = {
        "device": device,
        "device_type": device.type,
        "platform": platform.system(),
        "architecture": platform.machine(),
    }
    
    if device.type == "cuda":
        info.update({
            "device_name": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_reserved": torch.cuda.memory_reserved(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "cuda_version": torch.version.cuda,
            "supports_mixed_precision": True,
            "supports_torch_compile": True,
        })
    elif device.type == "mps":
        info.update({
            "device_name": "Apple Silicon MPS",
            "device_count": 1,
            "memory_total": "Unified Memory",
            "supports_mixed_precision": False,  # MPS has different precision handling
            "supports_torch_compile": False,    # May not be stable on MPS yet
            "mps_fallback_enabled": True,
        })
    else:  # CPU
        info.update({
            "device_name": f"CPU ({platform.processor()})",
            "device_count": torch.get_num_threads(),
            "supports_mixed_precision": False,
            "supports_torch_compile": True,
        })
    
    return info


def get_device_specific_config() -> Dict[str, Any]:
    """
    Get device-specific configuration for optimal performance.
    
    Returns:
        Dict containing device-optimized configuration settings
    """
    device_info = get_device_info()
    device_type = device_info["device_type"]
    
    if device_type == "cuda":
        return {
            "batch_size_multiplier": 1.0,
            "gradient_accumulation_steps": 1,
            "use_mixed_precision": True,
            "use_torch_compile": True,
            "memory_management": "cuda",
            "pin_memory": True,
            "num_workers": 4,
            "non_blocking": True,
        }
    elif device_type == "mps":
        return {
            "batch_size_multiplier": 0.8,  # MPS may need smaller batches
            "gradient_accumulation_steps": 2,  # Compensate with more accumulation
            "use_mixed_precision": False,  # MPS precision handling is different
            "use_torch_compile": False,    # May not be stable on MPS
            "memory_management": "mps",
            "pin_memory": False,           # Not beneficial for unified memory
            "num_workers": 2,              # Fewer workers for MPS
            "non_blocking": False,         # MPS prefers blocking transfers
        }
    else:  # CPU
        return {
            "batch_size_multiplier": 0.5,  # Smaller batches for CPU
            "gradient_accumulation_steps": 4,  # More accumulation for CPU
            "use_mixed_precision": False,
            "use_torch_compile": True,     # Can help CPU performance
            "memory_management": "cpu",
            "pin_memory": False,
            "num_workers": torch.get_num_threads() // 2,
            "non_blocking": False,
        }


def move_to_device(obj, device: Optional[torch.device] = None, **kwargs) -> Any:
    """
    Safely move tensors, models, or other objects to the specified device.
    Handles device-specific optimizations and fallbacks.
    
    Args:
        obj: Object to move (tensor, model, etc.)
        device: Target device (if None, uses optimal device)
        **kwargs: Additional arguments for .to() method
    
    Returns:
        Object moved to the target device
    """
    if device is None:
        device = get_optimal_device()
    
    try:
        if device.type == "mps":
            # MPS-specific handling
            return obj.to(device, non_blocking=False, **kwargs)
        elif device.type == "cuda":
            # CUDA-specific handling
            return obj.to(device, non_blocking=kwargs.get('non_blocking', True), **kwargs)
        else:
            # CPU handling
            return obj.to(device, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to move object to {device}, falling back to CPU: {e}")
        return obj.to(torch.device("cpu"), **kwargs)


def optimize_memory(device: Optional[torch.device] = None) -> None:
    """
    Perform device-specific memory optimization.
    
    Args:
        device: Device to optimize (if None, uses optimal device)
    """
    if device is None:
        device = get_optimal_device()
    
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        # CPU doesn't need explicit memory management
    except Exception as e:
        logger.warning(f"Failed to optimize memory for {device}: {e}")


def get_attention_implementation(device: Optional[torch.device] = None) -> str:
    """
    Get the best attention implementation for the given device.
    
    Args:
        device: Target device (if None, uses optimal device)
    
    Returns:
        str: Attention implementation to use ("flash_attn", "standard", "xformers")
    """
    if device is None:
        device = get_optimal_device()
    
    if device.type == "cuda":
        # Try to use flash attention on CUDA
        try:
            import flash_attn
            return "flash_attn"
        except ImportError:
            logger.info("flash_attn not available, using standard attention")
            return "standard"
    elif device.type == "mps":
        # MPS may not support flash attention, use standard
        return "standard"
    else:
        # CPU uses standard attention
        return "standard"


def setup_device_environment() -> Tuple[torch.device, Dict[str, Any]]:
    """
    Setup the complete device environment for ASI-Arch.
    
    Returns:
        Tuple of (device, config) for use throughout the application
    """
    device = get_optimal_device()
    device_info = get_device_info()
    config = get_device_specific_config()
    
    # Log device setup
    logger.info(f"Device setup complete:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Type: {device_info['device_type']}")
    logger.info(f"  Name: {device_info['device_name']}")
    logger.info(f"  Mixed Precision: {config['use_mixed_precision']}")
    logger.info(f"  Torch Compile: {config['use_torch_compile']}")
    
    # Set environment optimizations
    if device.type == "cuda":
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif device.type == "mps":
        # MPS optimizations
        try:
            torch.backends.mps.enable_fallback(True)
        except AttributeError:
            # Fallback setting not available in this PyTorch version
            pass
    
    return device, config


def check_device_compatibility() -> Dict[str, bool]:
    """
    Check compatibility of various operations with the current device.
    
    Returns:
        Dict of operation compatibility flags
    """
    device = get_optimal_device()
    compatibility = {
        "flash_attention": False,
        "torch_compile": False,
        "mixed_precision": False,
        "gradient_checkpointing": True,  # Usually supported everywhere
        "distributed_training": False,
    }
    
    try:
        if device.type == "cuda":
            compatibility.update({
                "flash_attention": True,
                "torch_compile": True,
                "mixed_precision": True,
                "distributed_training": torch.cuda.device_count() > 1,
            })
        elif device.type == "mps":
            compatibility.update({
                "torch_compile": False,  # May not be stable yet
                "mixed_precision": False,
                "distributed_training": False,
            })
        else:  # CPU
            compatibility.update({
                "torch_compile": True,
                "mixed_precision": False,
                "distributed_training": False,
            })
    except Exception as e:
        logger.warning(f"Error checking device compatibility: {e}")
    
    return compatibility


if __name__ == "__main__":
    # Test device detection
    logging.basicConfig(level=logging.INFO)
    
    device, config = setup_device_environment()
    print(f"\nDevice Setup Results:")
    print(f"Optimal Device: {device}")
    print(f"Configuration: {config}")
    
    compatibility = check_device_compatibility()
    print(f"\nCompatibility Check:")
    for feature, supported in compatibility.items():
        print(f"  {feature}: {'✓' if supported else '✗'}")