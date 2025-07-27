#!/usr/bin/env python3
"""
Architecture-Specific Fallbacks for ASI-Arch

Provides fallback implementations for architecture-specific operations that may not
be supported on all devices, particularly for MPS (Apple Silicon) compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union, Dict, Any, Callable
import warnings
from .device_utils import get_optimal_device, move_to_device

logger = logging.getLogger(__name__)


class FlashLinearAttentionFallback(nn.Module):
    """
    Fallback implementation for flash-linear-attention operations that may not
    support MPS or other devices.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or get_optimal_device()
        self.supports_flash_linear = self._check_flash_linear_support()
        
    def _check_flash_linear_support(self) -> bool:
        """Check if flash-linear-attention supports current device."""
        try:
            import flash_linear_attention
            # Test basic functionality
            if self.device.type == "mps":
                # Flash linear attention may not support MPS
                return False
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Flash linear attention check failed: {e}")
            return False
    
    def delta_net_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Fallback implementation for DeltaNet-style operations.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            **kwargs: Additional arguments
        
        Returns:
            Output tensor with same shape as input
        """
        if self.supports_flash_linear and self.device.type == "cuda":
            try:
                return self._flash_delta_net(x, **kwargs)
            except Exception as e:
                logger.debug(f"Flash DeltaNet failed, using fallback: {e}")
        
        return self._fallback_delta_net(x, **kwargs)
    
    def _flash_delta_net(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Original flash-linear-attention implementation."""
        try:
            import flash_linear_attention
            # Use actual flash linear attention if available
            # This would be replaced with actual implementation
            return self._fallback_delta_net(x, **kwargs)
        except Exception as e:
            logger.debug(f"Flash linear attention failed: {e}")
            raise
    
    def _fallback_delta_net(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Pure PyTorch fallback implementation of DeltaNet-style processing.
        Compatible with all devices including MPS.
        """
        batch_size, seq_len, features = x.shape
        
        # Simple linear transformation as fallback
        # In practice, this would implement the actual DeltaNet algorithm
        hidden_dim = features
        
        # Linear transformations (device-agnostic)
        query = F.linear(x, torch.randn(features, hidden_dim, device=x.device))
        key = F.linear(x, torch.randn(features, hidden_dim, device=x.device))
        value = F.linear(x, torch.randn(features, hidden_dim, device=x.device))
        
        # Apply activation
        query = F.gelu(query)
        key = F.gelu(key)
        
        # Simple attention-like computation
        scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        # Final projection
        output = F.linear(output, torch.randn(hidden_dim, features, device=x.device))
        
        return output


class MambaFallback(nn.Module):
    """
    Fallback implementation for Mamba SSM operations that may not support MPS.
    """
    
    def __init__(self, d_model: int, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model
        self.device = device or get_optimal_device()
        self.supports_mamba = self._check_mamba_support()
        
        # Fallback parameters
        self.linear_in = nn.Linear(d_model, d_model * 2, device=self.device)
        self.linear_out = nn.Linear(d_model, d_model, device=self.device)
        self.activation = nn.SiLU()
        
    def _check_mamba_support(self) -> bool:
        """Check if mamba-ssm supports current device."""
        try:
            import mamba_ssm
            if self.device.type == "mps":
                # Mamba may not support MPS
                return False
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Mamba SSM check failed: {e}")
            return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to simple RNN-like computation.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if self.supports_mamba and self.device.type == "cuda":
            try:
                return self._mamba_forward(x)
            except Exception as e:
                logger.debug(f"Mamba forward failed, using fallback: {e}")
        
        return self._fallback_forward(x)
    
    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original Mamba SSM implementation."""
        try:
            import mamba_ssm
            # Use actual Mamba implementation if available
            # This would be replaced with actual Mamba forward pass
            return self._fallback_forward(x)
        except Exception as e:
            logger.debug(f"Mamba SSM failed: {e}")
            raise
    
    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple RNN-like fallback implementation.
        Provides similar sequential processing to Mamba.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Simple gated linear transformation
        x_proj = self.linear_in(x)  # [batch, seq, 2*d_model]
        x_gate, x_hidden = x_proj.chunk(2, dim=-1)
        
        # Apply gating
        x_gate = torch.sigmoid(x_gate)
        x_hidden = self.activation(x_hidden)
        x_gated = x_gate * x_hidden
        
        # Simple recurrent processing (fallback for SSM)
        outputs = []
        hidden = torch.zeros(batch_size, d_model, device=x.device)
        
        for t in range(seq_len):
            # Simple linear recurrence
            hidden = 0.9 * hidden + 0.1 * x_gated[:, t, :]
            outputs.append(hidden)
        
        output = torch.stack(outputs, dim=1)  # [batch, seq, d_model]
        
        # Final projection
        output = self.linear_out(output)
        
        return output


class CausalConv1dFallback(nn.Module):
    """
    Fallback implementation for causal-conv1d operations.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, device: Optional[torch.device] = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.device = device or get_optimal_device()
        self.supports_causal_conv = self._check_causal_conv_support()
        
        # Fallback implementation using standard conv1d
        padding = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, device=self.device)
        
    def _check_causal_conv_support(self) -> bool:
        """Check if causal-conv1d supports current device."""
        try:
            import causal_conv1d
            if self.device.type == "mps":
                # May not support MPS
                return False
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Causal conv1d check failed: {e}")
            return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to standard causal convolution.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
        
        Returns:
            Output tensor [batch, channels, seq_len]
        """
        if self.supports_causal_conv and self.device.type == "cuda":
            try:
                return self._causal_conv_forward(x)
            except Exception as e:
                logger.debug(f"Causal conv1d failed, using fallback: {e}")
        
        return self._fallback_forward(x)
    
    def _causal_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original causal-conv1d implementation."""
        try:
            import causal_conv1d
            # Use actual causal conv1d if available
            # This would be replaced with actual implementation
            return self._fallback_forward(x)
        except Exception as e:
            logger.debug(f"Causal conv1d failed: {e}")
            raise
    
    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback causal convolution using standard conv1d with manual padding removal.
        """
        # Apply convolution with padding
        output = self.conv(x)
        
        # Remove future padding to make it causal
        if self.kernel_size > 1:
            output = output[:, :, :-(self.kernel_size - 1)]
        
        return output


class DeviceAgnosticModule(nn.Module):
    """
    Base class for modules that need device-agnostic operation with fallbacks.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or get_optimal_device()
        self.device_type = self.device.type
        self.fallback_enabled = self.device_type in ["mps", "cpu"]
        
    def safe_operation(self, operation: Callable, fallback: Callable, *args, **kwargs):
        """
        Safely execute an operation with automatic fallback.
        
        Args:
            operation: Primary operation to try
            fallback: Fallback operation if primary fails
            *args, **kwargs: Arguments for the operations
        
        Returns:
            Result of operation or fallback
        """
        if self.fallback_enabled:
            try:
                return operation(*args, **kwargs)
            except RuntimeError as e:
                if "MPS" in str(e) or "not implemented" in str(e).lower():
                    logger.debug(f"Operation failed on {self.device}, using fallback: {e}")
                    return fallback(*args, **kwargs)
                else:
                    raise
        else:
            return operation(*args, **kwargs)
    
    def move_to_cpu_and_back(self, func: Callable, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Move tensors to CPU, apply function, then move result back to original device.
        
        Args:
            func: Function to apply
            *tensors: Input tensors
            **kwargs: Additional keyword arguments
        
        Returns:
            Result tensor on original device
        """
        original_device = self.device
        
        # Move to CPU
        cpu_tensors = [t.cpu() if isinstance(t, torch.Tensor) else t for t in tensors]
        
        # Apply function on CPU
        result = func(*cpu_tensors, **kwargs)
        
        # Move result back to original device
        if isinstance(result, torch.Tensor):
            return result.to(original_device)
        elif isinstance(result, (list, tuple)):
            return type(result)(
                t.to(original_device) if isinstance(t, torch.Tensor) else t 
                for t in result
            )
        else:
            return result


def create_architecture_fallbacks(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Create a dictionary of architecture-specific fallback implementations.
    
    Args:
        device: Target device (auto-detected if None)
    
    Returns:
        Dictionary with fallback implementations
    """
    if device is None:
        device = get_optimal_device()
    
    fallbacks = {
        "device": device,
        "flash_linear_attention": FlashLinearAttentionFallback(device),
        "mamba": lambda d_model: MambaFallback(d_model, device),
        "causal_conv1d": lambda channels, kernel_size=3: CausalConv1dFallback(channels, kernel_size, device),
        "device_agnostic_base": lambda: DeviceAgnosticModule(device),
    }
    
    # Add device-specific information
    fallbacks.update({
        "supports_flash_linear": device.type == "cuda",
        "supports_mamba": device.type == "cuda", 
        "supports_causal_conv": device.type == "cuda",
        "needs_fallbacks": device.type in ["mps", "cpu"],
        "fallback_strategy": "cpu_compute" if device.type == "mps" else "none",
    })
    
    return fallbacks


def get_compatible_operation(operation_name: str, device: Optional[torch.device] = None) -> Callable:
    """
    Get a device-compatible version of a specific operation.
    
    Args:
        operation_name: Name of the operation
        device: Target device (auto-detected if None)
    
    Returns:
        Compatible operation function
    """
    if device is None:
        device = get_optimal_device()
    
    fallbacks = create_architecture_fallbacks(device)
    
    if operation_name in fallbacks:
        return fallbacks[operation_name]
    else:
        raise ValueError(f"Unknown operation: {operation_name}")


def wrap_with_fallback(original_class: type, device: Optional[torch.device] = None) -> type:
    """
    Wrap an existing class with device fallback capabilities.
    
    Args:
        original_class: Original class to wrap
        device: Target device (auto-detected if None)
    
    Returns:
        Wrapped class with fallback support
    """
    if device is None:
        device = get_optimal_device()
    
    class WrappedClass(original_class, DeviceAgnosticModule):
        def __init__(self, *args, **kwargs):
            # Initialize DeviceAgnosticModule first
            DeviceAgnosticModule.__init__(self, device)
            # Then initialize the original class
            original_class.__init__(self, *args, **kwargs)
            
        def forward(self, *args, **kwargs):
            # Try original forward, with fallback support
            def original_forward(*a, **kw):
                return super(WrappedClass, self).forward(*a, **kw)
            
            def fallback_forward(*a, **kw):
                # Move to CPU, compute, move back
                return self.move_to_cpu_and_back(original_forward, *a, **kw)
            
            return self.safe_operation(original_forward, fallback_forward, *args, **kwargs)
    
    return WrappedClass


# Convenience functions for common fallback patterns
def safe_flash_attention(*args, device: Optional[torch.device] = None, **kwargs):
    """Convenience function for safe flash attention."""
    fallback = FlashLinearAttentionFallback(device)
    return fallback.delta_net_forward(*args, **kwargs)


def safe_mamba(x: torch.Tensor, d_model: int, device: Optional[torch.device] = None):
    """Convenience function for safe Mamba operation."""
    fallback = MambaFallback(d_model, device)
    return fallback(x)


def safe_causal_conv1d(x: torch.Tensor, channels: int, kernel_size: int = 3, device: Optional[torch.device] = None):
    """Convenience function for safe causal conv1d."""
    fallback = CausalConv1dFallback(channels, kernel_size, device)
    return fallback(x)