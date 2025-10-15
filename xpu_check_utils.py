"""
Device utilities for memory monitoring and management across XPU, CUDA, and CPU.
Provides comprehensive diagnostics for Intel XPU and NVIDIA CUDA devices.
"""

import torch
import sys


def validate_xpu_setup():
    """
    Validate Intel XPU setup and availability.
    
    Returns:
        bool: True if XPU is properly configured and available, False otherwise.
    """
    try:
        import intel_extension_for_pytorch as ipex
        print(f"[DEBUG] Intel Extension for PyTorch version: {ipex.__version__}")
        
        if hasattr(torch, "xpu"):
            print(f"[DEBUG] torch.xpu module available: True")
            if torch.xpu.is_available():
                print(f"[DEBUG] XPU devices available: {torch.xpu.device_count()}")
                # Test basic XPU operation
                test_tensor = torch.randn(2, 2).to('xpu')
                print(f"[DEBUG] XPU test tensor created successfully on device: {test_tensor.device}")
                return True
            else:
                print("[DEBUG] torch.xpu.is_available() returned False")
        else:
            print("[DEBUG] torch.xpu module not available")
    except ImportError as e:
        print(f"[DEBUG] Intel Extension for PyTorch not available: {e}")
    except Exception as e:
        print(f"[DEBUG] XPU validation error: {e}")
    
    return False


def check_device_memory():
    """
    Check available memory on all devices (XPU, CUDA, and system RAM).
    Provides detailed memory statistics including total, allocated, reserved, and free memory.
    """
    print("\n" + "="*60)
    print("DEVICE MEMORY ANALYSIS")
    print("="*60)
    
    # Check XPU memory
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("🔍 XPU MEMORY STATUS:")
        for i in range(torch.xpu.device_count()):
            try:
                # Get device properties
                props = torch.xpu.get_device_properties(i)
                total_memory = props.total_memory
                
                # Get current memory usage
                allocated = torch.xpu.memory_allocated(i)
                reserved = torch.xpu.memory_reserved(i)
                free_memory = total_memory - reserved
                
                print(f"  XPU {i} ({props.name}):")
                print(f"    Total Memory:     {total_memory / 1024**3:.2f} GB")
                print(f"    Allocated:        {allocated / 1024**3:.2f} GB")
                print(f"    Reserved:         {reserved / 1024**3:.2f} GB")
                print(f"    Free:             {free_memory / 1024**3:.2f} GB")
                print(f"    Utilization:      {(reserved/total_memory)*100:.1f}%")
                
                # Test allocation
                try:
                    test_size = 100 * 1024 * 1024  # 100MB test
                    test_tensor = torch.randn(test_size // 4, device=f'xpu:{i}')
                    print(f"    Status:           ✅ Functional")
                    del test_tensor
                    torch.xpu.empty_cache()
                except Exception as e:
                    print(f"    Status:           ❌ Error: {e}")
                    
            except Exception as e:
                print(f"  XPU {i}: ❌ Error getting info: {e}")
    else:
        print("❌ XPU not available")
    
    # Check CUDA memory
    if torch.cuda.is_available():
        print("\n🔍 CUDA MEMORY STATUS:")
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                free_memory = total_memory - reserved
                
                print(f"  CUDA {i} ({props.name}):")
                print(f"    Total Memory:     {total_memory / 1024**3:.2f} GB")
                print(f"    Allocated:        {allocated / 1024**3:.2f} GB")
                print(f"    Reserved:         {reserved / 1024**3:.2f} GB")
                print(f"    Free:             {free_memory / 1024**3:.2f} GB")
                print(f"    Utilization:      {(reserved/total_memory)*100:.1f}%")
                
            except Exception as e:
                print(f"  CUDA {i}: ❌ Error: {e}")
    else:
        print("❌ CUDA not available")
    
    # Check system RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"\n🔍 SYSTEM RAM:")
        print(f"    Total:            {ram.total / 1024**3:.2f} GB")
        print(f"    Available:        {ram.available / 1024**3:.2f} GB")
        print(f"    Used:             {ram.used / 1024**3:.2f} GB")
        print(f"    Utilization:      {ram.percent:.1f}%")
    except ImportError:
        print("\n❌ psutil not available for RAM info")
    
    print("="*60 + "\n")


def estimate_model_memory_requirements():
    """
    Estimate memory requirements for the model.
    
    Returns:
        float: Total estimated memory requirement in GB.
    """
    print("\n" + "="*60)
    print("MODEL MEMORY ESTIMATION")
    print("="*60)
    
    # These are rough estimates based on typical model sizes
    estimates = {
        "Model Weights (fp16)": 2.5,  # GB
        "VAE": 0.8,  # GB
        "Conditioner": 0.5,  # GB
        "Working Memory": 1.0,  # GB for intermediate calculations
        "Safety Buffer": 0.5,  # GB
    }
    
    total_estimated = sum(estimates.values())
    
    print("📊 ESTIMATED MEMORY REQUIREMENTS:")
    for component, size in estimates.items():
        print(f"    {component:<20}: {size:.1f} GB")
    print(f"    {'='*20}")
    print(f"    {'Total Estimated':<20}: {total_estimated:.1f} GB")
    print("="*60 + "\n")
    
    return total_estimated


def quick_memory_check():
    """Quick memory status check without detailed analysis."""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            try:
                props = torch.xpu.get_device_properties(i)
                total = props.total_memory / 1024**3
                reserved = torch.xpu.memory_reserved(i) / 1024**3
                free = total - reserved
                print(f"XPU {i}: {free:.2f}GB free / {total:.2f}GB total")
            except:
                print(f"XPU {i}: Status unknown")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = total - reserved
                print(f"CUDA {i}: {free:.2f}GB free / {total:.2f}GB total")
            except:
                print(f"CUDA {i}: Status unknown")

def cleanup_xpu_memory():
    """Clean up XPU memory on all devices"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            try:
                torch.xpu.empty_cache()
                # torch.xpu.synchronize()
                print(f"[DEBUG] Cleaned XPU {i} memory")
            except Exception as e:
                print(f"[DEBUG] Failed to clean XPU {i}: {e}")

def get_optimal_device_with_memory_check():
    """
    Automatically detect the best available device with memory verification.
    
    Returns:
        str: Device string ('xpu', 'cuda', 'mps', or 'cpu').
    """
    print("\n" + "="*60)
    print("AUTO-DETECTING OPTIMAL DEVICE")
    print("="*60)
    
    # Check XPU first
    if validate_xpu_setup():
        try:
            props = torch.xpu.get_device_properties(0)
            free_memory = (props.total_memory - torch.xpu.memory_reserved(0)) / 1024**3
            if free_memory >= 4.0:  # At least 4GB free
                print(f"✅ Selected XPU (Free: {free_memory:.2f}GB)")
                return 'xpu'
            else:
                print(f"⚠️ XPU has insufficient free memory ({free_memory:.2f}GB)")
        except Exception as e:
            print(f"⚠️ XPU memory check failed: {e}")
    
    # Check CUDA
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            free_memory = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            if free_memory >= 4.0:
                print(f"✅ Selected CUDA (Free: {free_memory:.2f}GB)")
                return 'cuda'
            else:
                print(f"⚠️ CUDA has insufficient free memory ({free_memory:.2f}GB)")
        except Exception as e:
            print(f"⚠️ CUDA memory check failed: {e}")
    
    # Default to CPU
    print("✅ Selected CPU (fallback)")
    return 'cpu'


def clear_device_cache(device):
    """
    Clear memory cache for the specified device.
    
    Args:
        device (str): Device type ('xpu', 'cuda', etc.)
    """
    if device == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.empty_cache()
        print("[DEBUG] 🧹 Cleared XPU cache")
    elif device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[DEBUG] 🧹 Cleared CUDA cache")


def get_memory_callback(device):
    """
    Get a memory monitoring callback for the specified device.
    
    Args:
        device (str): Device type ('xpu', 'cuda', etc.)
        
    Returns:
        callable: Function that prints current memory usage when called.
    """
    def xpu_callback():
        allocated = torch.xpu.memory_allocated(0) / 1024**3
        reserved = torch.xpu.memory_reserved(0) / 1024**3
        print(f"    📊 XPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def cuda_callback():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"    📊 CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def null_callback():
        pass
    
    if device == 'xpu':
        return xpu_callback
    elif device == 'cuda':
        return cuda_callback
    else:
        return null_callback
    
def debug_pipeline_tensors(pipeline, device):
    """Debug tensor types and devices in pipeline"""
    print(f"\n🔍 PIPELINE TENSOR DEBUG:")
    
    # Check model components
    for name, component in [('model', pipeline.model), ('vae', pipeline.vae), ('conditioner', pipeline.conditioner)]:
        if hasattr(component, 'parameters'):
            try:
                param = next(component.parameters())
                print(f"  {name}: device={param.device}, dtype={param.dtype}")
            except StopIteration:
                print(f"  {name}: No parameters found")
    
    # Test a small forward pass
    try:
        print(f"  Testing small forward pass...")
        test_input = torch.randn(1, 3, 256, 256, dtype=torch.float16, device=device)
        with torch.no_grad():
            cond_output = pipeline.conditioner(image=test_input)
            print(f"  ✅ Conditioner test passed: {type(cond_output)}")
            
            # Check output tensor info
            if isinstance(cond_output, dict) and 'main' in cond_output:
                main_output = cond_output['main']
                print(f"  Conditioner output - device: {main_output.device}, dtype: {main_output.dtype}, shape: {main_output.shape}")
            
    except Exception as e:
        print(f"  ❌ Conditioner test failed: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")

def force_xpu_fp16_consistency(pipeline, device):
    """Force consistent FP16 for XPU pipeline"""
    print("[DEBUG] 🔧 Forcing FP16 consistency for XPU...")
    
    try:
        # Ensure all components use FP16
        pipeline.model = pipeline.model.half().to(device)
        pipeline.vae = pipeline.vae.half().to(device)
        pipeline.conditioner = pipeline.conditioner.half().to(device)
        
        # Set pipeline dtype
        pipeline.dtype = torch.float16
        
        # Ensure device is set correctly
        pipeline.device = torch.device(device)
        
        print("[DEBUG] ✅ Applied FP16 consistency")
        
        # Verify the changes
        for name, component in [('model', pipeline.model), ('vae', pipeline.vae), ('conditioner', pipeline.conditioner)]:
            if hasattr(component, 'parameters'):
                try:
                    param = next(component.parameters())
                    print(f"  {name}: device={param.device}, dtype={param.dtype}")
                except StopIteration:
                    print(f"  {name}: No parameters to check")
        
        return True
        
    except Exception as e:
        print(f"[DEBUG] ❌ Failed to apply FP16 consistency: {e}")
        return False

def validate_xpu_model_loading(pipeline, device):
    """Validate that model loaded correctly on XPU"""
    print(f"\n🔍 XPU MODEL VALIDATION:")
    
    try:
        # Test each component
        test_image = torch.randn(1, 3, 256, 256, dtype=torch.float16, device=device)
        
        # Test conditioner
        with torch.no_grad():
            cond_out = pipeline.conditioner(image=test_image)
            if isinstance(cond_out, dict) and 'main' in cond_out:
                main_shape = cond_out['main'].shape
                print(f"  ✅ Conditioner: Output shape {main_shape}")
            else:
                print(f"  ✅ Conditioner: Output type {type(cond_out)}")
        
        # Test a single diffusion step
        latents = torch.randn(1, *pipeline.vae.latent_shape, dtype=torch.float16, device=device)
        timestep = torch.tensor([500], device=device, dtype=torch.float16)
        
        with torch.no_grad():
            noise_pred = pipeline.model(latents, timestep, cond_out)
            print(f"  ✅ Model: Output shape {noise_pred.shape}")
        
        # Test VAE decode (small test)
        small_latents = torch.randn(1, pipeline.vae.latent_shape[0], 32, 32, 32, dtype=torch.float16, device=device)
        with torch.no_grad():
            try:
                vae_out = pipeline.vae.decode(small_latents)
                print(f"  ✅ VAE: Decode test passed")
            except Exception as e:
                print(f"  ⚠️ VAE: Decode test failed: {e}")
        
        print(f"  ✅ XPU model validation passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ XPU model validation failed: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        return False