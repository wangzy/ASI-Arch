#!/usr/bin/env python3
"""
MPS-Specific Memory Management for ASI-Arch

Provides intelligent memory management for Apple Silicon MPS devices,
handling unified memory architecture and MPS-specific optimizations.
"""

import torch
import logging
import gc
import psutil
import threading
import time
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager
from .device_utils import get_optimal_device

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Intelligent memory manager with device-specific optimizations.
    Handles CUDA, MPS, and CPU memory management strategies.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_optimal_device()
        self.device_type = self.device.type
        self.memory_stats = {}
        self.cleanup_threshold = self._get_cleanup_threshold()
        self.monitoring_enabled = False
        self._monitor_thread = None
        
        logger.info(f"MemoryManager initialized for {self.device}")
        logger.info(f"Cleanup threshold: {self.cleanup_threshold}")
    
    def _get_cleanup_threshold(self) -> float:
        """Get memory cleanup threshold based on device type."""
        if self.device_type == "cuda":
            return 0.85  # Cleanup at 85% GPU memory usage
        elif self.device_type == "mps":
            return 0.80  # More conservative for MPS (unified memory)
        else:
            return 0.90  # CPU can handle higher memory usage
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information for the current device."""
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "timestamp": time.time(),
        }
        
        try:
            if self.device_type == "cuda":
                info.update(self._get_cuda_memory_info())
            elif self.device_type == "mps":
                info.update(self._get_mps_memory_info())
            else:
                info.update(self._get_cpu_memory_info())
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            info["error"] = str(e)
        
        return info
    
    def _get_cuda_memory_info(self) -> Dict[str, Any]:
        """Get CUDA-specific memory information."""
        device_idx = self.device.index or 0
        
        return {
            "allocated": torch.cuda.memory_allocated(device_idx),
            "reserved": torch.cuda.memory_reserved(device_idx),
            "max_allocated": torch.cuda.max_memory_allocated(device_idx),
            "max_reserved": torch.cuda.max_memory_reserved(device_idx),
            "total_memory": torch.cuda.get_device_properties(device_idx).total_memory,
            "memory_usage_percent": torch.cuda.memory_allocated(device_idx) / torch.cuda.get_device_properties(device_idx).total_memory * 100,
        }
    
    def _get_mps_memory_info(self) -> Dict[str, Any]:
        """Get MPS-specific memory information."""
        try:
            return {
                "current_allocated": torch.mps.current_allocated_memory(),
                "driver_allocated": torch.mps.driver_allocated_memory(),
                "system_memory": psutil.virtual_memory().total,
                "system_available": psutil.virtual_memory().available,
                "system_usage_percent": psutil.virtual_memory().percent,
                "unified_memory": True,
                "memory_pressure": self._get_mps_memory_pressure(),
            }
        except Exception as e:
            logger.debug(f"MPS memory info failed: {e}")
            return self._get_cpu_memory_info()
    
    def _get_cpu_memory_info(self) -> Dict[str, Any]:
        """Get CPU memory information."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "usage_percent": memory.percent,
            "swap_total": psutil.swap_memory().total,
            "swap_used": psutil.swap_memory().used,
        }
    
    def _get_mps_memory_pressure(self) -> str:
        """Estimate MPS memory pressure based on system metrics."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return "high"
            elif memory.percent > 75:
                return "medium"
            else:
                return "low"
        except:
            return "unknown"
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform device-specific memory cleanup.
        
        Args:
            force: Force cleanup regardless of current usage
        
        Returns:
            Dictionary with cleanup results
        """
        memory_before = self.get_memory_info()
        
        # Check if cleanup is needed
        if not force and not self._should_cleanup(memory_before):
            return {
                "cleanup_performed": False,
                "reason": "cleanup not needed",
                "memory_before": memory_before,
                "memory_after": memory_before,
            }
        
        logger.info(f"Performing memory cleanup on {self.device}")
        
        # Python garbage collection first
        gc.collect()
        
        # Device-specific cleanup
        if self.device_type == "cuda":
            self._cleanup_cuda()
        elif self.device_type == "mps":
            self._cleanup_mps()
        else:
            self._cleanup_cpu()
        
        memory_after = self.get_memory_info()
        
        result = {
            "cleanup_performed": True,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_freed": self._calculate_memory_freed(memory_before, memory_after),
        }
        
        logger.info(f"Memory cleanup completed: {result['memory_freed']} freed")
        return result
    
    def _should_cleanup(self, memory_info: Dict[str, Any]) -> bool:
        """Determine if memory cleanup is needed based on current usage."""
        if self.device_type == "cuda":
            usage_percent = memory_info.get("memory_usage_percent", 0)
            return usage_percent > (self.cleanup_threshold * 100)
        elif self.device_type == "mps":
            usage_percent = memory_info.get("system_usage_percent", 0)
            pressure = memory_info.get("memory_pressure", "low")
            return usage_percent > (self.cleanup_threshold * 100) or pressure == "high"
        else:  # CPU
            usage_percent = memory_info.get("usage_percent", 0)
            return usage_percent > (self.cleanup_threshold * 100)
    
    def _cleanup_cuda(self):
        """CUDA-specific memory cleanup."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
    
    def _cleanup_mps(self):
        """MPS-specific memory cleanup."""
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except Exception as e:
            logger.debug(f"MPS cleanup failed: {e}")
        
        # Additional system-level cleanup for unified memory
        gc.collect()
    
    def _cleanup_cpu(self):
        """CPU memory cleanup."""
        # For CPU, just run garbage collection
        gc.collect()
        
        # Try to encourage OS to free unused memory
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):
                # Windows
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            elif hasattr(ctypes, 'CDLL'):
                # Unix-like systems
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass
        except:
            pass
    
    def _calculate_memory_freed(self, before: Dict[str, Any], after: Dict[str, Any]) -> str:
        """Calculate how much memory was freed during cleanup."""
        try:
            if self.device_type == "cuda":
                freed = before.get("allocated", 0) - after.get("allocated", 0)
                return f"{freed / 1024**2:.1f} MB GPU"
            elif self.device_type == "mps":
                freed = before.get("current_allocated", 0) - after.get("current_allocated", 0)
                return f"{freed / 1024**2:.1f} MB MPS"
            else:
                freed = before.get("used", 0) - after.get("used", 0)
                return f"{freed / 1024**2:.1f} MB RAM"
        except:
            return "unknown"
    
    @contextmanager
    def memory_monitor(self, cleanup_on_exit: bool = True):
        """
        Context manager for monitoring memory usage during operations.
        
        Args:
            cleanup_on_exit: Whether to cleanup memory when exiting context
        """
        start_memory = self.get_memory_info()
        logger.debug(f"Memory monitor started: {start_memory}")
        
        try:
            yield self
        finally:
            end_memory = self.get_memory_info()
            logger.debug(f"Memory monitor ended: {end_memory}")
            
            if cleanup_on_exit:
                self.cleanup_memory()
    
    def start_monitoring(self, interval: float = 30.0):
        """
        Start background memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_enabled:
            logger.warning("Memory monitoring already enabled")
            return
        
        self.monitoring_enabled = True
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _memory_monitor_loop(self, interval: float):
        """Background memory monitoring loop."""
        while self.monitoring_enabled:
            try:
                memory_info = self.get_memory_info()
                self.memory_stats[time.time()] = memory_info
                
                # Check if cleanup is needed
                if self._should_cleanup(memory_info):
                    logger.info("Automatic memory cleanup triggered")
                    self.cleanup_memory()
                
                # Keep only last hour of stats
                cutoff_time = time.time() - 3600
                self.memory_stats = {
                    t: stats for t, stats in self.memory_stats.items() 
                    if t > cutoff_time
                }
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_memory_trends(self) -> Dict[str, Any]:
        """Get memory usage trends from monitoring data."""
        if not self.memory_stats:
            return {"error": "No monitoring data available"}
        
        times = sorted(self.memory_stats.keys())
        if len(times) < 2:
            return {"error": "Insufficient data for trends"}
        
        # Calculate trends based on device type
        if self.device_type == "cuda":
            key = "allocated"
        elif self.device_type == "mps":
            key = "current_allocated"
        else:
            key = "used"
        
        values = [self.memory_stats[t].get(key, 0) for t in times]
        
        # Simple trend analysis
        recent_avg = sum(values[-5:]) / min(5, len(values))
        overall_avg = sum(values) / len(values)
        
        trend = "stable"
        if recent_avg > overall_avg * 1.1:
            trend = "increasing"
        elif recent_avg < overall_avg * 0.9:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "current_usage": values[-1],
            "average_usage": overall_avg,
            "peak_usage": max(values),
            "min_usage": min(values),
            "data_points": len(values),
            "time_span_hours": (times[-1] - times[0]) / 3600,
        }


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(device: Optional[torch.device] = None) -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _global_memory_manager
    
    target_device = device or get_optimal_device()
    
    if (_global_memory_manager is None or 
        _global_memory_manager.device != target_device):
        _global_memory_manager = MemoryManager(target_device)
    
    return _global_memory_manager


# Convenience functions
def cleanup_memory(device: Optional[torch.device] = None, force: bool = False) -> Dict[str, Any]:
    """Cleanup memory on the specified device."""
    manager = get_memory_manager(device)
    return manager.cleanup_memory(force)


def get_memory_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get memory information for the specified device."""
    manager = get_memory_manager(device)
    return manager.get_memory_info()


@contextmanager
def memory_monitor(device: Optional[torch.device] = None, cleanup_on_exit: bool = True):
    """Context manager for memory monitoring."""
    manager = get_memory_manager(device)
    with manager.memory_monitor(cleanup_on_exit):
        yield manager


def start_memory_monitoring(device: Optional[torch.device] = None, interval: float = 30.0):
    """Start background memory monitoring."""
    manager = get_memory_manager(device)
    manager.start_monitoring(interval)


def stop_memory_monitoring(device: Optional[torch.device] = None):
    """Stop background memory monitoring."""
    manager = get_memory_manager(device)
    manager.stop_monitoring()


def optimize_for_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Optimize model for device-specific memory patterns.
    
    Args:
        model: PyTorch model to optimize
        device: Target device (auto-detected if None)
    
    Returns:
        Optimized model
    """
    if device is None:
        device = get_optimal_device()
    
    # Move model to device
    model = model.to(device)
    
    # Device-specific optimizations
    if device.type == "mps":
        # MPS-specific optimizations
        logger.info("Applying MPS optimizations")
        
        # Disable some features that may not work well on MPS
        for module in model.modules():
            if hasattr(module, 'memory_efficient'):
                module.memory_efficient = True
            if hasattr(module, 'use_checkpointing'):
                module.use_checkpointing = True
    
    elif device.type == "cuda":
        # CUDA-specific optimizations
        logger.info("Applying CUDA optimizations")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    else:  # CPU
        # CPU-specific optimizations
        logger.info("Applying CPU optimizations")
        
        # Enable CPU optimizations
        torch.set_num_threads(min(8, torch.get_num_threads()))
    
    return model