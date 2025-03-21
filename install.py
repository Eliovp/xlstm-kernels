#!/usr/bin/env python3
"""
Installation script for xlstm-kernels and AMD-optimized xlstm library.
Run this script with: python install.py
"""

import os
import sys
import subprocess
import shutil
import platform

def main():
    print("====== Starting installation of xlstm-kernels and AMD-optimized xlstm ======")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    try:
        # Step 1: Install xlstm-kernels package
        print("\n\n==== Step 1: Installing xlstm-kernels package ====")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        
        # Step 2: Clone xlstm repository
        print("\n\n==== Step 2: Cloning AMD-optimized xlstm repository ====")
        temp_dir = os.path.join(current_dir, "xlstm_temp")
        if os.path.exists(temp_dir):
            print(f"Removing existing directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        
        print(f"Cloning xlstm into: {temp_dir}")
        subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", temp_dir])
        
        # Step 3: Install xlstm package
        print("\n\n==== Step 3: Installing AMD-optimized xlstm package ====")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", temp_dir])
        
        print("\n\n====== Installation completed successfully! ======")
        print("You can now use the AMD-optimized xlstm library.")
        print("Example usage:")
        print("```python")
        print("import xlstm")
        print("import mlstm_kernels")
        print("from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge")
        print("```")
        
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 