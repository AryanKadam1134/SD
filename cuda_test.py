import torch
import sys
import subprocess
import numpy as np

def verify_cuda_setup():
    print("=== CUDA Setup Verification ===")
    
    # Version Information
    print("\nVersion Information:")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Check NVIDIA Driver
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        print("\nNVIDIA Driver Status:")
        print(nvidia_smi)
    except Exception as e:
        print("❌ NVIDIA Driver not found or not responding")
        print(f"Error: {e}")
        return False

    # Check PyTorch CUDA
    print("\nPyTorch CUDA Status:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA functionality with error handling
        try:
            # Test with NumPy array conversion
            np_array = np.array([1.0, 2.0, 3.0])
            torch_tensor = torch.from_numpy(np_array).cuda()
            print("\n✓ NumPy to CUDA tensor conversion successful")
            
            # Test direct tensor creation
            cuda_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print("✓ Direct CUDA tensor creation successful")
            
            # Test basic operations
            result = cuda_tensor * 2
            print("✓ CUDA tensor operations successful")
            
            # Cleanup
            del torch_tensor, cuda_tensor, result
            torch.cuda.empty_cache()
            print("✓ CUDA memory cleanup successful")
            
            return True
        except Exception as e:
            print(f"\n❌ CUDA test failed: {str(e)}")
            return False
    else:
        print("❌ CUDA is not available")
        return False

if __name__ == "__main__":
    success = verify_cuda_setup()
    if success:
        print("\n✅ CUDA setup is complete and working properly")
    else:
        print("\n❌ CUDA setup needs attention")