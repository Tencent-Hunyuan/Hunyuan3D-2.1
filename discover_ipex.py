#!/usr/bin/env python3
import intel_extension_for_pytorch as ipex
import torch
import inspect

def explore_module(module, name, max_depth=2, current_depth=0):
    """Recursively explore a module"""
    if current_depth >= max_depth:
        return
    
    print(f"{'  ' * current_depth}=== {name} ===")
    
    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue
            
        try:
            attr = getattr(module, attr_name)
            attr_type = type(attr).__name__
            
            if callable(attr):
                try:
                    sig = inspect.signature(attr)
                    print(f"{'  ' * current_depth}{name}.{attr_name}{sig}")
                except:
                    print(f"{'  ' * current_depth}{name}.{attr_name}() - {attr_type}")
            elif hasattr(attr, '__module__') and 'intel' in str(attr.__module__):
                print(f"{'  ' * current_depth}{name}.{attr_name} - {attr_type} (submodule)")
                if current_depth < max_depth - 1:
                    explore_module(attr, f"{name}.{attr_name}", max_depth, current_depth + 1)
            else:
                print(f"{'  ' * current_depth}{name}.{attr_name} - {attr_type}")
                
        except Exception as e:
            print(f"{'  ' * current_depth}{name}.{attr_name} - Error: {e}")

def main():
    print("Intel Extension for PyTorch API Discovery")
    print("=" * 50)
    print(f"IPEX Version: {ipex.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Explore main IPEX module
    explore_module(ipex, "ipex", max_depth=2)
    
    # Explore XPU module
    print("\n" + "=" * 50)
    explore_module(torch.xpu, "torch.xpu", max_depth=2)
    
    # Check for common optimization functions
    print("\n" + "=" * 50)
    print("=== Common IPEX Functions ===")
    
    common_functions = [
        'optimize', 'enable_onednn_fusion', 'disable_onednn_fusion',
        'set_fp32_math_mode', 'get_fp32_math_mode',
        'enable_auto_mixed_precision', 'disable_auto_mixed_precision'
    ]
    
    for func_name in common_functions:
        if hasattr(ipex, func_name):
            func = getattr(ipex, func_name)
            try:
                sig = inspect.signature(func)
                print(f"✓ ipex.{func_name}{sig}")
            except:
                print(f"✓ ipex.{func_name}()")
        else:
            print(f"✗ ipex.{func_name} - Not available")
    
    # Check XPU specific functions
    print("\n=== XPU Device Functions ===")
    xpu_functions = [
        'device_count', 'current_device', 'set_device', 'get_device_name',
        'empty_cache', 'synchronize', 'memory_stats', 'memory_summary',
        'get_device_properties', 'is_available'
    ]
    
    for func_name in xpu_functions:
        if hasattr(torch.xpu, func_name):
            func = getattr(torch.xpu, func_name)
            try:
                sig = inspect.signature(func)
                print(f"✓ torch.xpu.{func_name}{sig}")
            except:
                print(f"✓ torch.xpu.{func_name}()")
        else:
            print(f"✗ torch.xpu.{func_name} - Not available")

if __name__ == "__main__":
    main()