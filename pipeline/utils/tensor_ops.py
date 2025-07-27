#!/usr/bin/env python3
"""
Device-Agnostic Tensor Operations for ASI-Arch

Provides tensor operations that work correctly across CUDA, MPS (Apple Silicon), and CPU devices.
Handles device-specific limitations and provides fallback mechanisms for unsupported operations.
"""

import torch
import logging
from typing import Optional, Tuple, Any, Union
import warnings
from .device_utils import get_optimal_device, move_to_device, get_attention_implementation

logger = logging.getLogger(__name__)


def safe_attention(query: torch.Tensor, 
                  key: torch.Tensor, 
                  value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  dropout_p: float = 0.0,
                  is_causal: bool = False,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe attention computation with automatic fallback for unsupported operations.
    
    Args:
        query, key, value: Attention tensors
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention
        device: Target device (auto-detected if None)
    
    Returns:
        Attention output tensor
    """
    if device is None:
        device = get_optimal_device()
    
    # Move tensors to device
    query = move_to_device(query, device)
    key = move_to_device(key, device)
    value = move_to_device(value, device)
    if mask is not None:
        mask = move_to_device(mask, device)
    
    attention_impl = get_attention_implementation(device)
    
    try:
        if attention_impl == "flash_attn" and device.type == "cuda":
            # Try flash attention on CUDA
            import flash_attn
            from flash_attn import flash_attn_func
            
            # Reshape for flash attention if needed
            if query.dim() == 4:  # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)  
                value = value.transpose(1, 2)
            
            output = flash_attn_func(query, key, value, dropout_p, causal=is_causal)
            
            # Reshape back if needed
            if output.dim() == 4:
                output = output.transpose(1, 2)
            
            return output
            
    except (ImportError, RuntimeError, NotImplementedError) as e:
        logger.debug(f"Flash attention failed on {device}, falling back to standard: {e}")
    
    # Standard attention implementation (works on all devices)
    return standard_attention(query, key, value, mask, dropout_p, is_causal)


def standard_attention(query: torch.Tensor,
                      key: torch.Tensor, 
                      value: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      dropout_p: float = 0.0,
                      is_causal: bool = False) -> torch.Tensor:
    """
    Standard attention implementation that works on all devices.
    """
    # Compute attention scores
    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    if is_causal:
        seq_len = scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0.0 and query.training:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, value)
    
    return output


def safe_linear_attention(query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         feature_map_fn: callable = None,
                         device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe linear attention computation.
    
    Args:
        query, key, value: Attention tensors
        feature_map_fn: Feature mapping function (e.g., for linear attention)
        device: Target device (auto-detected if None)
    
    Returns:
        Linear attention output
    """
    if device is None:
        device = get_optimal_device()
    
    # Move tensors to device
    query = move_to_device(query, device)
    key = move_to_device(key, device)
    value = move_to_device(value, device)
    
    # Apply feature mapping if provided
    if feature_map_fn is not None:
        query = feature_map_fn(query)
        key = feature_map_fn(key)
    
    # Linear attention computation: (Q * K^T) * V
    # More memory efficient: Q * (K^T * V)
    try:
        # Try efficient computation
        kv = torch.matmul(key.transpose(-2, -1), value)
        output = torch.matmul(query, kv)
        return output
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            # MPS fallback: move to CPU for computation
            logger.debug(f"MPS operation failed, falling back to CPU: {e}")
            query_cpu = query.cpu()
            key_cpu = key.cpu()
            value_cpu = value.cpu()
            
            kv_cpu = torch.matmul(key_cpu.transpose(-2, -1), value_cpu)
            output_cpu = torch.matmul(query_cpu, kv_cpu)
            
            return output_cpu.to(device)
        else:
            raise


def safe_einsum(equation: str, *tensors: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe einsum operation with fallback for unsupported operations.
    
    Args:
        equation: Einsum equation string
        *tensors: Input tensors
        device: Target device (auto-detected if None)
    
    Returns:
        Einsum result tensor
    """
    if device is None:
        device = get_optimal_device()
    
    # Move all tensors to device
    tensors = [move_to_device(t, device) for t in tensors]
    
    try:
        return torch.einsum(equation, *tensors)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            # MPS fallback: move to CPU for computation
            logger.debug(f"MPS einsum failed, falling back to CPU: {e}")
            tensors_cpu = [t.cpu() for t in tensors]
            result_cpu = torch.einsum(equation, *tensors_cpu)
            return result_cpu.to(device)
        else:
            raise


def safe_bmm(input: torch.Tensor, mat2: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe batch matrix multiplication.
    
    Args:
        input: First batch of matrices
        mat2: Second batch of matrices  
        device: Target device (auto-detected if None)
    
    Returns:
        Batch matrix multiplication result
    """
    if device is None:
        device = get_optimal_device()
    
    input = move_to_device(input, device)
    mat2 = move_to_device(mat2, device)
    
    try:
        return torch.bmm(input, mat2)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            # MPS fallback
            logger.debug(f"MPS bmm failed, falling back to CPU: {e}")
            input_cpu = input.cpu()
            mat2_cpu = mat2.cpu()
            result_cpu = torch.bmm(input_cpu, mat2_cpu)
            return result_cpu.to(device)
        else:
            raise


def safe_layer_norm(input: torch.Tensor, 
                   normalized_shape: Union[int, Tuple[int, ...]],
                   weight: Optional[torch.Tensor] = None,
                   bias: Optional[torch.Tensor] = None,
                   eps: float = 1e-5,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe layer normalization.
    
    Args:
        input: Input tensor
        normalized_shape: Input shape from an expected input
        weight: Optional learnable weights
        bias: Optional learnable bias
        eps: Value added to denominator for numerical stability
        device: Target device (auto-detected if None)
    
    Returns:
        Layer normalized tensor
    """
    if device is None:
        device = get_optimal_device()
    
    input = move_to_device(input, device)
    if weight is not None:
        weight = move_to_device(weight, device)
    if bias is not None:
        bias = move_to_device(bias, device)
    
    try:
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            # MPS fallback
            logger.debug(f"MPS layer_norm failed, falling back to CPU: {e}")
            input_cpu = input.cpu()
            weight_cpu = weight.cpu() if weight is not None else None
            bias_cpu = bias.cpu() if bias is not None else None
            
            result_cpu = torch.nn.functional.layer_norm(input_cpu, normalized_shape, weight_cpu, bias_cpu, eps)
            return result_cpu.to(device)
        else:
            raise


def safe_gelu(input: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Device-safe GELU activation.
    
    Args:
        input: Input tensor
        device: Target device (auto-detected if None)
    
    Returns:
        GELU activated tensor
    """
    if device is None:
        device = get_optimal_device()
    
    input = move_to_device(input, device)
    
    try:
        return torch.nn.functional.gelu(input)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            # MPS fallback: use approximate GELU
            logger.debug(f"MPS GELU failed, using approximate GELU: {e}")
            return input * 0.5 * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (input + 0.044715 * torch.pow(input, 3))))
        else:
            raise


def create_device_agnostic_model(model_class: type, *args, device: Optional[torch.device] = None, **kwargs) -> torch.nn.Module:
    """
    Create a model instance that's properly configured for the target device.
    
    Args:
        model_class: Model class to instantiate
        *args: Positional arguments for model constructor
        device: Target device (auto-detected if None)
        **kwargs: Keyword arguments for model constructor
    
    Returns:
        Model instance configured for the target device
    """
    if device is None:
        device = get_optimal_device()
    
    # Create model
    model = model_class(*args, **kwargs)
    
    # Move to device
    model = move_to_device(model, device)
    
    # Device-specific optimizations
    if device.type == "cuda":
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif device.type == "mps":
        # Enable MPS optimizations
        torch.backends.mps.enable_fallback(True)
    
    return model


def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """
    Get memory statistics for the current device.
    
    Args:
        device: Device to check (auto-detected if None)
    
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = get_optimal_device()
    
    stats = {"device": device.type}
    
    try:
        if device.type == "cuda":
            stats.update({
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "max_allocated": torch.cuda.max_memory_allocated(device),
                "max_reserved": torch.cuda.max_memory_reserved(device),
            })
        elif device.type == "mps":
            stats.update({
                "allocated": torch.mps.current_allocated_memory(),
                "driver_allocated": torch.mps.driver_allocated_memory(),
            })
        else:  # CPU
            stats.update({
                "allocated": "N/A (CPU)",
                "reserved": "N/A (CPU)",
            })
    except Exception as e:
        logger.warning(f"Failed to get memory stats for {device}: {e}")
        stats["error"] = str(e)
    
    return stats


# Convenience functions for common operations
def safe_matmul(a: torch.Tensor, b: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Device-safe matrix multiplication."""
    if device is None:
        device = get_optimal_device()
    
    a = move_to_device(a, device)
    b = move_to_device(b, device)
    
    try:
        return torch.matmul(a, b)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            logger.debug(f"MPS matmul failed, falling back to CPU: {e}")
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            result_cpu = torch.matmul(a_cpu, b_cpu)
            return result_cpu.to(device)
        else:
            raise


def safe_softmax(input: torch.Tensor, dim: int = -1, device: Optional[torch.device] = None) -> torch.Tensor:
    """Device-safe softmax computation."""
    if device is None:
        device = get_optimal_device()
    
    input = move_to_device(input, device)
    
    try:
        return torch.softmax(input, dim=dim)
    except RuntimeError as e:
        if "MPS" in str(e) or device.type == "mps":
            logger.debug(f"MPS softmax failed, falling back to CPU: {e}")
            input_cpu = input.cpu()
            result_cpu = torch.softmax(input_cpu, dim=dim)
            return result_cpu.to(device)
        else:
            raise