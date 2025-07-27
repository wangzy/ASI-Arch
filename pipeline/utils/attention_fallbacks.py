#!/usr/bin/env python3
"""
Flash Attention and Architecture-Specific Fallbacks for ASI-Arch

Provides compatibility layers for flash attention and other architecture-specific operations
that may not be supported on all devices (especially MPS/Apple Silicon).
"""

import torch
import logging
from typing import Optional, Tuple, Union, Dict, Any
import warnings
from .device_utils import get_optimal_device, move_to_device

logger = logging.getLogger(__name__)


class FlashAttentionFallback:
    """
    Flash Attention implementation with automatic fallbacks for different devices.
    Provides the same interface as flash_attn but works on CUDA, MPS, and CPU.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_optimal_device()
        self.flash_attn_available = self._check_flash_attn_availability()
        self.use_flash_attn = self.flash_attn_available and self.device.type == "cuda"
        
        logger.info(f"FlashAttentionFallback initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Flash attention available: {self.flash_attn_available}")
        logger.info(f"  Using flash attention: {self.use_flash_attn}")
    
    def _check_flash_attn_availability(self) -> bool:
        """Check if flash_attn is available and functional."""
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            logger.debug("flash_attn not available")
            return False
        except Exception as e:
            logger.debug(f"flash_attn import failed: {e}")
            return False
    
    def attention(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 dropout_p: float = 0.0,
                 is_causal: bool = False,
                 scale: Optional[float] = None) -> torch.Tensor:
        """
        Unified attention interface with automatic fallback.
        
        Args:
            query, key, value: Attention tensors [batch, seq_len, num_heads, head_dim]
            mask: Optional attention mask
            dropout_p: Dropout probability
            is_causal: Whether to use causal attention
            scale: Attention scaling factor (auto-computed if None)
        
        Returns:
            Attention output tensor
        """
        # Move tensors to device
        query = move_to_device(query, self.device)
        key = move_to_device(key, self.device)
        value = move_to_device(value, self.device)
        if mask is not None:
            mask = move_to_device(mask, self.device)
        
        # Try flash attention first if available and on CUDA
        if self.use_flash_attn:
            try:
                return self._flash_attention(query, key, value, dropout_p, is_causal)
            except Exception as e:
                logger.warning(f"Flash attention failed, falling back to standard: {e}")
                self.use_flash_attn = False  # Disable for future calls
        
        # Use optimized attention for the specific device
        if self.device.type == "mps":
            return self._mps_optimized_attention(query, key, value, mask, dropout_p, is_causal, scale)
        elif self.device.type == "cuda":
            return self._cuda_optimized_attention(query, key, value, mask, dropout_p, is_causal, scale)
        else:
            return self._cpu_optimized_attention(query, key, value, mask, dropout_p, is_causal, scale)
    
    def _flash_attention(self,
                        query: torch.Tensor,
                        key: torch.Tensor, 
                        value: torch.Tensor,
                        dropout_p: float,
                        is_causal: bool) -> torch.Tensor:
        """Flash attention implementation for CUDA."""
        from flash_attn import flash_attn_func
        
        # Ensure correct tensor format for flash attention
        # Expected: [batch, seq_len, num_heads, head_dim]
        if query.dim() == 4 and query.size(1) != query.size(-2):
            # Convert from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            need_transpose_back = True
        else:
            need_transpose_back = False
        
        output = flash_attn_func(query, key, value, dropout_p, causal=is_causal)
        
        # Convert back if needed
        if need_transpose_back:
            output = output.transpose(1, 2)
        
        return output
    
    def _mps_optimized_attention(self,
                                query: torch.Tensor,
                                key: torch.Tensor,
                                value: torch.Tensor,
                                mask: Optional[torch.Tensor],
                                dropout_p: float,
                                is_causal: bool,
                                scale: Optional[float]) -> torch.Tensor:
        """MPS-optimized attention with fallback mechanisms."""
        try:
            return self._standard_attention(query, key, value, mask, dropout_p, is_causal, scale)
        except RuntimeError as e:
            if "MPS" in str(e):
                logger.debug(f"MPS attention failed, falling back to CPU: {e}")
                # Move to CPU, compute, then move back
                query_cpu = query.cpu()
                key_cpu = key.cpu()
                value_cpu = value.cpu()
                mask_cpu = mask.cpu() if mask is not None else None
                
                result_cpu = self._standard_attention(query_cpu, key_cpu, value_cpu, mask_cpu, dropout_p, is_causal, scale)
                return result_cpu.to(self.device)
            else:
                raise
    
    def _cuda_optimized_attention(self,
                                 query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 mask: Optional[torch.Tensor],
                                 dropout_p: float,
                                 is_causal: bool,
                                 scale: Optional[float]) -> torch.Tensor:
        """CUDA-optimized attention implementation."""
        # Use torch's scaled_dot_product_attention if available (PyTorch 2.0+)
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                attn_mask = mask
                if is_causal:
                    # scaled_dot_product_attention handles causal masking internally
                    attn_mask = None if mask is None else mask
                
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, 
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal
                )
        except Exception as e:
            logger.debug(f"CUDA scaled_dot_product_attention failed: {e}")
        
        # Fallback to standard attention
        return self._standard_attention(query, key, value, mask, dropout_p, is_causal, scale)
    
    def _cpu_optimized_attention(self,
                                query: torch.Tensor,
                                key: torch.Tensor,
                                value: torch.Tensor,
                                mask: Optional[torch.Tensor],
                                dropout_p: float,
                                is_causal: bool,
                                scale: Optional[float]) -> torch.Tensor:
        """CPU-optimized attention implementation."""
        # For CPU, use chunked computation to reduce memory usage
        return self._chunked_attention(query, key, value, mask, dropout_p, is_causal, scale, chunk_size=1024)
    
    def _standard_attention(self,
                           query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           mask: Optional[torch.Tensor],
                           dropout_p: float,
                           is_causal: bool,
                           scale: Optional[float]) -> torch.Tensor:
        """Standard attention implementation that works on all devices."""
        if scale is None:
            scale = query.size(-1) ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if is_causal:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0 and query.training:
            attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def _chunked_attention(self,
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          mask: Optional[torch.Tensor],
                          dropout_p: float,
                          is_causal: bool,
                          scale: Optional[float],
                          chunk_size: int = 1024) -> torch.Tensor:
        """Memory-efficient chunked attention computation."""
        seq_len = query.size(-2)
        
        if seq_len <= chunk_size:
            return self._standard_attention(query, key, value, mask, dropout_p, is_causal, scale)
        
        # Process in chunks
        outputs = []
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            query_chunk = query[..., i:end_i, :]
            
            # For causal attention, we only need keys/values up to current position
            if is_causal:
                key_chunk = key[..., :end_i, :]
                value_chunk = value[..., :end_i, :]
                mask_chunk = mask[..., i:end_i, :end_i] if mask is not None else None
            else:
                key_chunk = key
                value_chunk = value
                mask_chunk = mask[..., i:end_i, :] if mask is not None else None
            
            chunk_output = self._standard_attention(
                query_chunk, key_chunk, value_chunk, mask_chunk, dropout_p, is_causal, scale
            )
            outputs.append(chunk_output)
        
        return torch.cat(outputs, dim=-2)


class LinearAttentionFallback:
    """
    Linear attention implementations with device-specific optimizations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_optimal_device()
    
    def linear_attention(self,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        feature_map: Optional[callable] = None,
                        eps: float = 1e-6) -> torch.Tensor:
        """
        Linear attention computation: O(N) complexity.
        
        Computes attention as: softmax(Q)^T @ (softmax(K)^T @ V)
        
        Args:
            query, key, value: Attention tensors
            feature_map: Optional feature mapping function
            eps: Small value for numerical stability
        
        Returns:
            Linear attention output
        """
        # Move tensors to device
        query = move_to_device(query, self.device)
        key = move_to_device(key, self.device)
        value = move_to_device(value, self.device)
        
        # Apply feature mapping if provided
        if feature_map is not None:
            query = feature_map(query)
            key = feature_map(key)
        else:
            # Default: positive feature mapping
            query = torch.nn.functional.elu(query) + 1
            key = torch.nn.functional.elu(key) + 1
        
        try:
            if self.device.type == "mps":
                return self._mps_linear_attention(query, key, value, eps)
            else:
                return self._standard_linear_attention(query, key, value, eps)
        except RuntimeError as e:
            if "MPS" in str(e) or self.device.type == "mps":
                logger.debug(f"MPS linear attention failed, falling back to CPU: {e}")
                query_cpu = query.cpu()
                key_cpu = key.cpu()
                value_cpu = value.cpu()
                result_cpu = self._standard_linear_attention(query_cpu, key_cpu, value_cpu, eps)
                return result_cpu.to(self.device)
            else:
                raise
    
    def _standard_linear_attention(self,
                                  query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor,
                                  eps: float) -> torch.Tensor:
        """Standard linear attention implementation."""
        # Compute K^T @ V first (more memory efficient)
        kv = torch.matmul(key.transpose(-2, -1), value)
        
        # Compute Q @ (K^T @ V)
        output = torch.matmul(query, kv)
        
        # Normalization
        key_sum = key.sum(dim=-2, keepdim=True)
        normalizer = torch.matmul(query, key_sum.transpose(-2, -1))
        output = output / (normalizer + eps)
        
        return output
    
    def _mps_linear_attention(self,
                             query: torch.Tensor,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             eps: float) -> torch.Tensor:
        """MPS-specific linear attention with optimizations."""
        # MPS might have issues with some operations, so we break it down
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Process each head separately to avoid large matrix operations
        outputs = []
        for h in range(num_heads):
            q_h = query[:, :, h, :]  # [batch, seq, head_dim]
            k_h = key[:, :, h, :]
            v_h = value[:, :, h, :]
            
            # Compute attention for this head
            kv_h = torch.matmul(k_h.transpose(-2, -1), v_h)
            output_h = torch.matmul(q_h, kv_h)
            
            # Normalization
            key_sum_h = k_h.sum(dim=-2, keepdim=True)
            normalizer_h = torch.matmul(q_h, key_sum_h.transpose(-2, -1))
            output_h = output_h / (normalizer_h + eps)
            
            outputs.append(output_h.unsqueeze(2))  # Add head dimension back
        
        return torch.cat(outputs, dim=2)


def get_attention_backend(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get the appropriate attention backend for the given device.
    
    Args:
        device: Target device (auto-detected if None)
    
    Returns:
        Dictionary with attention implementations and configurations
    """
    if device is None:
        device = get_optimal_device()
    
    flash_fallback = FlashAttentionFallback(device)
    linear_fallback = LinearAttentionFallback(device)
    
    backend = {
        "device": device,
        "flash_attention": flash_fallback,
        "linear_attention": linear_fallback,
        "supports_flash": flash_fallback.use_flash_attn,
        "memory_efficient": device.type in ["cpu", "mps"],
    }
    
    # Device-specific recommendations
    if device.type == "cuda":
        backend.update({
            "recommended_batch_size": "large",
            "use_chunking": False,
            "compile_compatible": True,
        })
    elif device.type == "mps":
        backend.update({
            "recommended_batch_size": "medium", 
            "use_chunking": True,
            "compile_compatible": False,
        })
    else:  # CPU
        backend.update({
            "recommended_batch_size": "small",
            "use_chunking": True,
            "compile_compatible": True,
        })
    
    return backend


# Convenience functions for easy integration
def attention_with_fallback(query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           dropout_p: float = 0.0,
                           is_causal: bool = False,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    High-level attention function with automatic device-specific optimization.
    """
    flash_fallback = FlashAttentionFallback(device)
    return flash_fallback.attention(query, key, value, mask, dropout_p, is_causal)


def linear_attention_with_fallback(query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor,
                                  feature_map: Optional[callable] = None,
                                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    High-level linear attention function with automatic device-specific optimization.
    """
    linear_fallback = LinearAttentionFallback(device)
    return linear_fallback.linear_attention(query, key, value, feature_map)