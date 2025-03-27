import torch
import platform
import os
import sys
import subprocess

def get_cuda_info():
    """Get CUDA information from system"""
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        return nvidia_smi
    except:
        return "nvidia-smi not available"

def test_setup():
    # System information
    print("=== System Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.release()}")
    
    # CUDA System Information
    print("\n=== CUDA System Information ===")
    print(get_cuda_info())
    
    # PyTorch CUDA Information
    print("\n=== PyTorch CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**2:.0f}MB")
        
        # Progressive GPU tests
        print("\n=== GPU Memory Tests ===")
        try:
            # Small tensor test
            x = torch.rand(100, 100).cuda()
            print("✓ Small tensor test passed")
            
            # Medium tensor test
            y = torch.rand(500, 500).cuda()
            print("✓ Medium tensor test passed")
            
            # Cleanup
            del x, y
            torch.cuda.empty_cache()
            print("✓ Memory cleanup successful")
            
        except RuntimeError as e:
            print(f"❌ GPU test failed: {e}")
    else:
        print("❌ CUDA not available - using CPU only")
        print("Please ensure CUDA Toolkit 11.8 is properly installed")

if __name__ == "__main__":
    test_setup()