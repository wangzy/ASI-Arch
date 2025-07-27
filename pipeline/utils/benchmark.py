#!/usr/bin/env python3
"""
Performance Benchmarking Utilities for ASI-Arch

Provides comprehensive benchmarking tools to compare performance across
CUDA, MPS (Apple Silicon), and CPU devices for architecture discovery.
"""

import torch
import time
import gc
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import statistics
from .device_utils import get_optimal_device, get_device_info
from .memory_manager import get_memory_manager
from .attention_fallbacks import FlashAttentionFallback, LinearAttentionFallback
from .tensor_ops import safe_matmul, safe_attention, safe_linear_attention

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structure for benchmark results."""
    operation: str
    device: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_used: int
    memory_peak: int
    iterations: int
    success_rate: float
    error_count: int
    notes: str = ""


class DeviceBenchmark:
    """
    Comprehensive benchmarking suite for different devices.
    """
    
    def __init__(self, device: Optional[torch.device] = None, warmup_iterations: int = 3):
        self.device = device or get_optimal_device()
        self.device_info = get_device_info()
        self.memory_manager = get_memory_manager(self.device)
        self.warmup_iterations = warmup_iterations
        self.results = []
        
        logger.info(f"DeviceBenchmark initialized for {self.device}")
        logger.info(f"Device info: {self.device_info['device_name']}")
    
    @contextmanager
    def benchmark_context(self, operation: str):
        """Context manager for benchmarking operations."""
        # Memory cleanup before benchmark
        self.memory_manager.cleanup_memory()
        
        # Get initial memory
        initial_memory = self.memory_manager.get_memory_info()
        
        try:
            yield
        finally:
            # Final memory cleanup
            self.memory_manager.cleanup_memory()
            final_memory = self.memory_manager.get_memory_info()
            
            logger.debug(f"Benchmark '{operation}' memory: "
                        f"initial={initial_memory}, final={final_memory}")
    
    def time_operation(self, func: Callable, iterations: int = 10, 
                      warmup: Optional[int] = None) -> Tuple[List[float], int, int]:
        """
        Time an operation with proper synchronization and error handling.
        
        Args:
            func: Function to benchmark
            iterations: Number of timing iterations
            warmup: Warmup iterations (uses self.warmup_iterations if None)
        
        Returns:
            Tuple of (execution_times, success_count, error_count)
        """
        if warmup is None:
            warmup = self.warmup_iterations
        
        # Warmup
        for _ in range(warmup):
            try:
                func()
                self._synchronize_device()
            except Exception:
                pass  # Ignore warmup errors
        
        # Clear any warmup effects
        self.memory_manager.cleanup_memory()
        
        # Timing runs
        times = []
        success_count = 0
        error_count = 0
        
        for i in range(iterations):
            try:
                start_time = time.time()
                result = func()
                self._synchronize_device()
                end_time = time.time()
                
                times.append(end_time - start_time)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Benchmark iteration {i} failed: {e}")
        
        return times, success_count, error_count
    
    def _synchronize_device(self):
        """Synchronize device for accurate timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def benchmark_operation(self, operation: str, func: Callable, 
                           data_size: int, iterations: int = 10) -> BenchmarkResult:
        """
        Benchmark a specific operation.
        
        Args:
            operation: Name of the operation
            func: Function to benchmark
            data_size: Size of data processed (for throughput calculation)
            iterations: Number of iterations
        
        Returns:
            BenchmarkResult with timing and memory statistics
        """
        with self.benchmark_context(operation):
            # Get initial memory
            memory_before = self.memory_manager.get_memory_info()
            
            # Time the operation
            times, success_count, error_count = self.time_operation(func, iterations)
            
            # Get final memory
            memory_after = self.memory_manager.get_memory_info()
            
            if not times:
                # All iterations failed
                return BenchmarkResult(
                    operation=operation,
                    device=str(self.device),
                    mean_time=float('inf'),
                    std_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    throughput=0.0,
                    memory_used=0,
                    memory_peak=0,
                    iterations=iterations,
                    success_rate=0.0,
                    error_count=error_count,
                    notes="All iterations failed"
                )
            
            # Calculate statistics
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            min_time = min(times)
            max_time = max(times)
            
            # Calculate throughput (operations per second)
            throughput = data_size / mean_time if mean_time > 0 else 0.0
            
            # Calculate memory usage
            memory_used = self._calculate_memory_diff(memory_before, memory_after)
            memory_peak = memory_used  # Simplified - could track peak during operation
            
            success_rate = success_count / iterations if iterations > 0 else 0.0
            
            result = BenchmarkResult(
                operation=operation,
                device=str(self.device),
                mean_time=mean_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                throughput=throughput,
                memory_used=memory_used,
                memory_peak=memory_peak,
                iterations=iterations,
                success_rate=success_rate,
                error_count=error_count
            )
            
            self.results.append(result)
            return result
    
    def _calculate_memory_diff(self, before: Dict[str, Any], after: Dict[str, Any]) -> int:
        """Calculate memory difference between two snapshots."""
        try:
            if self.device.type == "cuda":
                return after.get("allocated", 0) - before.get("allocated", 0)
            elif self.device.type == "mps":
                return after.get("current_allocated", 0) - before.get("current_allocated", 0)
            else:  # CPU
                return after.get("used", 0) - before.get("used", 0)
        except Exception:
            return 0
    
    def benchmark_attention_suite(self, batch_size: int = 2, seq_len: int = 512, 
                                 hidden_dim: int = 512, num_heads: int = 8) -> List[BenchmarkResult]:
        """
        Benchmark attention operations with different implementations.
        
        Args:
            batch_size: Batch size for attention
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        
        Returns:
            List of benchmark results for different attention types
        """
        head_dim = hidden_dim // num_heads
        results = []
        
        # Create test tensors
        def create_tensors():
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
            return q, k, v
        
        # Standard attention
        def standard_attention():
            q, k, v = create_tensors()
            return safe_attention(q, k, v, device=self.device)
        
        result = self.benchmark_operation(
            "standard_attention", standard_attention, 
            batch_size * seq_len * seq_len, iterations=10
        )
        results.append(result)
        
        # Flash attention (if available)
        flash_fallback = FlashAttentionFallback(self.device)
        if flash_fallback.use_flash_attn:
            def flash_attention():
                q, k, v = create_tensors()
                return flash_fallback.attention(q, k, v)
            
            result = self.benchmark_operation(
                "flash_attention", flash_attention,
                batch_size * seq_len * seq_len, iterations=10
            )
            results.append(result)
        
        # Linear attention
        def linear_attention():
            q, k, v = create_tensors()
            return safe_linear_attention(q, k, v, device=self.device)
        
        result = self.benchmark_operation(
            "linear_attention", linear_attention,
            batch_size * seq_len * hidden_dim, iterations=10
        )
        results.append(result)
        
        return results
    
    def benchmark_matrix_operations(self, sizes: List[int] = None) -> List[BenchmarkResult]:
        """
        Benchmark matrix operations at different sizes.
        
        Args:
            sizes: List of matrix sizes to test
        
        Returns:
            List of benchmark results for matrix operations
        """
        if sizes is None:
            sizes = [128, 256, 512, 1024]
        
        results = []
        
        for size in sizes:
            # Matrix multiplication
            def matmul_op():
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)
                return safe_matmul(a, b, device=self.device)
            
            result = self.benchmark_operation(
                f"matmul_{size}x{size}", matmul_op,
                size * size * size, iterations=10
            )
            results.append(result)
            
            # Batch matrix multiplication
            def bmm_op():
                a = torch.randn(32, size, size, device=self.device)
                b = torch.randn(32, size, size, device=self.device)
                return torch.bmm(a, b)
            
            result = self.benchmark_operation(
                f"bmm_{size}x{size}", bmm_op,
                32 * size * size * size, iterations=10
            )
            results.append(result)
        
        return results
    
    def benchmark_memory_operations(self) -> List[BenchmarkResult]:
        """Benchmark memory-intensive operations."""
        results = []
        
        # Large tensor creation
        def large_tensor_creation():
            return torch.randn(1000, 1000, device=self.device)
        
        result = self.benchmark_operation(
            "large_tensor_creation", large_tensor_creation,
            1000 * 1000, iterations=5
        )
        results.append(result)
        
        # Memory transfer (if not CPU)
        if self.device.type != "cpu":
            def memory_transfer():
                cpu_tensor = torch.randn(1000, 1000)
                device_tensor = cpu_tensor.to(self.device)
                return device_tensor.cpu()
            
            result = self.benchmark_operation(
                "memory_transfer", memory_transfer,
                1000 * 1000, iterations=5
            )
            results.append(result)
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Returns:
            Dictionary with all benchmark results and summary
        """
        logger.info(f"Starting full benchmark on {self.device}")
        
        all_results = []
        
        # Attention benchmarks
        logger.info("Running attention benchmarks...")
        attention_results = self.benchmark_attention_suite()
        all_results.extend(attention_results)
        
        # Matrix operation benchmarks
        logger.info("Running matrix operation benchmarks...")
        matrix_results = self.benchmark_matrix_operations()
        all_results.extend(matrix_results)
        
        # Memory operation benchmarks
        logger.info("Running memory operation benchmarks...")
        memory_results = self.benchmark_memory_operations()
        all_results.extend(memory_results)
        
        # Calculate summary statistics
        successful_results = [r for r in all_results if r.success_rate > 0.5]
        
        summary = {
            "device": str(self.device),
            "device_info": self.device_info,
            "total_operations": len(all_results),
            "successful_operations": len(successful_results),
            "success_rate": len(successful_results) / len(all_results) if all_results else 0,
            "average_throughput": statistics.mean([r.throughput for r in successful_results]) if successful_results else 0,
            "total_errors": sum(r.error_count for r in all_results),
        }
        
        benchmark_data = {
            "summary": summary,
            "results": [asdict(r) for r in all_results],
            "device_config": self.device_info,
        }
        
        logger.info(f"Benchmark completed: {summary}")
        return benchmark_data
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        benchmark_data = self.run_full_benchmark()
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")


class CrossDeviceBenchmark:
    """
    Compare performance across multiple devices.
    """
    
    def __init__(self, devices: Optional[List[torch.device]] = None):
        if devices is None:
            devices = self._detect_available_devices()
        
        self.devices = devices
        self.benchmarks = {device: DeviceBenchmark(device) for device in devices}
        
        logger.info(f"CrossDeviceBenchmark initialized for devices: {devices}")
    
    def _detect_available_devices(self) -> List[torch.device]:
        """Detect all available devices."""
        devices = [torch.device("cpu")]
        
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        
        return devices
    
    def run_comparison(self) -> Dict[str, Any]:
        """
        Run benchmarks on all devices and compare results.
        
        Returns:
            Comparison results across devices
        """
        logger.info("Running cross-device comparison...")
        
        device_results = {}
        
        for device in self.devices:
            logger.info(f"Benchmarking {device}...")
            device_results[str(device)] = self.benchmarks[device].run_full_benchmark()
        
        # Generate comparison
        comparison = self._generate_comparison(device_results)
        
        return {
            "device_results": device_results,
            "comparison": comparison,
        }
    
    def _generate_comparison(self, device_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics across devices."""
        devices = list(device_results.keys())
        
        if len(devices) < 2:
            return {"note": "Need at least 2 devices for comparison"}
        
        # Find common operations
        common_ops = set()
        for device, data in device_results.items():
            ops = {r["operation"] for r in data["results"] if r["success_rate"] > 0.5}
            if not common_ops:
                common_ops = ops
            else:
                common_ops &= ops
        
        # Compare performance for common operations
        operation_comparison = {}
        
        for op in common_ops:
            op_data = {}
            
            for device, data in device_results.items():
                # Find result for this operation
                for result in data["results"]:
                    if result["operation"] == op and result["success_rate"] > 0.5:
                        op_data[device] = {
                            "mean_time": result["mean_time"],
                            "throughput": result["throughput"],
                            "memory_used": result["memory_used"],
                        }
                        break
            
            if len(op_data) >= 2:
                operation_comparison[op] = op_data
        
        # Find fastest device for each operation
        fastest_device = {}
        for op, data in operation_comparison.items():
            if data:
                fastest = min(data.items(), key=lambda x: x[1]["mean_time"])
                fastest_device[op] = fastest[0]
        
        return {
            "common_operations": list(common_ops),
            "operation_comparison": operation_comparison,
            "fastest_device_per_operation": fastest_device,
            "device_rankings": self._rank_devices(device_results),
        }
    
    def _rank_devices(self, device_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank devices by overall performance."""
        device_scores = []
        
        for device, data in device_results.items():
            successful_results = [r for r in data["results"] if r["success_rate"] > 0.5]
            
            if successful_results:
                avg_throughput = statistics.mean([r["throughput"] for r in successful_results])
                success_rate = data["summary"]["success_rate"]
                
                # Simple scoring: throughput * success_rate
                score = avg_throughput * success_rate
                
                device_scores.append({
                    "device": device,
                    "score": score,
                    "avg_throughput": avg_throughput,
                    "success_rate": success_rate,
                })
        
        # Sort by score (descending)
        device_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return device_scores


# Convenience functions
def quick_benchmark(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Run a quick benchmark on the specified device."""
    benchmark = DeviceBenchmark(device)
    return benchmark.run_full_benchmark()


def compare_devices() -> Dict[str, Any]:
    """Compare performance across all available devices."""
    comparison = CrossDeviceBenchmark()
    return comparison.run_comparison()


def benchmark_attention(batch_size: int = 2, seq_len: int = 512, 
                       device: Optional[torch.device] = None) -> List[BenchmarkResult]:
    """Quick attention benchmark."""
    benchmark = DeviceBenchmark(device)
    return benchmark.benchmark_attention_suite(batch_size, seq_len)