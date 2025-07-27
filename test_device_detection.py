#!/usr/bin/env python3
"""
Test script for device detection functionality.
Run this to verify device detection works correctly after installing dependencies.
"""

import sys
import platform

def test_system_info():
    """Test basic system information detection."""
    print("=== System Information ===")
    print(f"Platform: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if this is Apple Silicon
    is_apple_silicon = (
        platform.system() == "Darwin" and 
        platform.machine() in ["arm64", "aarch64"]
    )
    print(f"Apple Silicon: {'Yes' if is_apple_silicon else 'No'}")
    print()

def test_pytorch_installation():
    """Test PyTorch installation and device detection."""
    try:
        import torch
        print("=== PyTorch Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test MPS availability (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f"MPS available: {mps_available}")
        
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        print()
        return True
    except ImportError:
        print("PyTorch not installed. Install with:")
        print("pip install torch>=2.6.0")
        return False

def test_device_utils():
    """Test ASI-Arch device utilities."""
    try:
        sys.path.append('./pipeline')
        from utils.device_utils import (
            setup_device_environment, 
            check_device_compatibility,
            get_device_info
        )
        
        print("=== ASI-Arch Device Detection ===")
        device, config = setup_device_environment()
        
        print(f"Optimal device: {device}")
        print(f"Device type: {device.type}")
        
        device_info = get_device_info()
        print(f"Device name: {device_info['device_name']}")
        print(f"Supports mixed precision: {device_info.get('supports_mixed_precision', False)}")
        print(f"Supports torch compile: {device_info.get('supports_torch_compile', False)}")
        
        print("\nDevice Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nCompatibility Check:")
        compatibility = check_device_compatibility()
        for feature, supported in compatibility.items():
            status = "✓" if supported else "✗"
            print(f"  {feature}: {status}")
        
        print()
        return True
        
    except ImportError as e:
        print(f"Device utils not available: {e}")
        return False

def test_tensor_operations():
    """Test device-agnostic tensor operations."""
    try:
        import torch
        sys.path.append('./pipeline')
        from utils.tensor_ops import (
            safe_matmul,
            safe_softmax,
            get_memory_stats,
            create_device_agnostic_model
        )
        
        print("=== Tensor Operations Test ===")
        
        # Create test tensors
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        
        # Test matrix multiplication
        result = safe_matmul(a, b)
        print(f"Matrix multiplication: {result.shape} on {result.device}")
        
        # Test softmax
        x = torch.randn(2, 10)
        softmax_result = safe_softmax(x)
        print(f"Softmax: {softmax_result.shape} on {softmax_result.device}")
        
        # Test memory stats
        memory_stats = get_memory_stats()
        print(f"Memory stats: {memory_stats}")
        
        print("Tensor operations test: PASSED")
        print()
        return True
        
    except Exception as e:
        print(f"Tensor operations test failed: {e}")
        return False


def test_mps_compatibility():
    """Test MPS compatibility checker."""
    try:
        import torch
        sys.path.append('./pipeline')
        from utils.error_handler import check_mps_compatibility
        
        print("=== MPS Compatibility Test ===")
        
        compatibility = check_mps_compatibility()
        print(f"Device: {compatibility['device']}")
        
        if compatibility.get('mps_specific'):
            print(f"MPS tests passed: {compatibility['all_tests_passed']}")
            if compatibility['unsupported_operations']:
                print(f"Unsupported operations: {compatibility['unsupported_operations']}")
            if compatibility['recommendations']:
                print("Recommendations:")
                for rec in compatibility['recommendations']:
                    print(f"  - {rec}")
        else:
            print("Not an MPS device")
        
        print("MPS compatibility test: PASSED")
        print()
        return True
        
    except Exception as e:
        print(f"MPS compatibility test failed: {e}")
        return False


def test_benchmarking():
    """Test performance benchmarking utilities."""
    try:
        import torch
        sys.path.append('./pipeline')
        from utils.benchmark import DeviceBenchmark, quick_benchmark
        
        print("=== Benchmarking Test ===")
        
        # Quick benchmark test
        print("Running quick benchmark (limited iterations)...")
        benchmark = DeviceBenchmark()
        
        # Test a simple operation
        def simple_op():
            a = torch.randn(10, 10, device=benchmark.device)
            return a @ a
        
        result = benchmark.benchmark_operation("simple_matmul", simple_op, 1000, iterations=3)
        print(f"Simple matmul benchmark:")
        print(f"  Mean time: {result.mean_time:.4f}s")
        print(f"  Success rate: {result.success_rate:.2f}")
        print(f"  Device: {result.device}")
        
        print("Benchmarking test: PASSED")
        print()
        return True
        
    except Exception as e:
        print(f"Benchmarking test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ASI-Arch Device Detection Test Suite")
    print("====================================")
    
    # Test basic system info
    test_system_info()
    
    # Test PyTorch
    pytorch_ok = test_pytorch_installation()
    
    if pytorch_ok:
        # Test device utilities
        device_utils_ok = test_device_utils()
        
        if device_utils_ok:
            # Test tensor operations
            tensor_ops_ok = test_tensor_operations()
            
            # Test MPS compatibility
            test_mps_compatibility()
            
            # Test benchmarking
            if tensor_ops_ok:
                test_benchmarking()
    
    print("Test complete!")
    print("\nNext steps:")
    if not pytorch_ok:
        print("1. Install PyTorch: pip install torch>=2.6.0")
    else:
        print("1. ✓ PyTorch installed")
    
    print("2. Install full ASI-Arch dependencies: pip install -r requirements.txt")
    print("3. Test the full pipeline: cd pipeline && python pipeline.py")

if __name__ == "__main__":
    main()