from setuptools import setup, find_packages
import subprocess
import os
import sys
import shutil
import pkg_resources
from setuptools.command.install import install
from setuptools.command.develop import develop

def check_transformers_installed():
    """Check if transformers is already installed and which version."""
    try:
        transformers_pkg = pkg_resources.get_distribution("transformers")
        print(f"Found existing transformers installation: {transformers_pkg.version}")
        return True, transformers_pkg.version
    except pkg_resources.DistributionNotFound:
        print("No existing transformers installation found.")
        return False, None

def install_transformers_fork():
    """Install the specific transformers fork."""
    print("====== Installing xLSTM-compatible transformers fork... ======")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/NX-AI/transformers.git@integrate_xlstm"
        ])
        print("xLSTM-compatible transformers fork installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install transformers fork: {e}")
        print("Will attempt to continue with installation anyway")
        return False

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        print("Starting xlstm-kernels installation...")
        try:
            # First run the standard install
            install.run(self)
            # Check and install transformers fork
            installed, version = check_transformers_installed()
            if not installed or input(f"Transformers {version} is already installed. Replace with xLSTM fork? (y/n): ").lower() == 'y':
                install_transformers_fork()
            # Then install xlstm
            self.install_xlstm()
        except Exception as e:
            print(f"Error during installation: {e}")
            raise

    def install_xlstm(self):
        print("====== Installing AMD-optimized xLSTM library... ======")
        try:
            # Get current directory
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
            
            # Create a temporary directory
            temp_dir = os.path.join(current_dir, "xlstm_temp")
            print(f"Temp directory will be: {temp_dir}")
            
            # Remove if it exists
            if os.path.exists(temp_dir):
                print(f"Removing existing temp directory: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove directory: {e}")
                    print("Attempting alternative removal method...")
                    
                    # Try force removal with system commands
                    import time
                    time.sleep(1)  # Wait briefly for any processes to release files
                    
                    try:
                        if os.name == 'nt':  # Windows
                            subprocess.check_call(f'rmdir /S /Q "{temp_dir}"', shell=True)
                        else:  # Linux/Mac
                            subprocess.check_call(f'rm -rf "{temp_dir}"', shell=True)
                    except Exception as e2:
                        print(f"Warning: Alternative removal also failed: {e2}")
                        print("Will attempt to continue with installation anyway")
            
            # Double-check dir is gone before proceeding
            if os.path.exists(temp_dir):
                print(f"Warning: {temp_dir} still exists. Installation may fail.")
                
            print("Cloning xlstm repository...")
            subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", temp_dir])
            
            print("Installing xlstm...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", temp_dir])
            
            print("AMD-optimized xLSTM library installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
            raise
        except Exception as e:
            print(f"Error installing xLSTM: {e}")
            raise

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        print("Starting xlstm-kernels development installation...")
        try:
            # First run the standard develop
            develop.run(self)
            # Check and install transformers fork
            installed, version = check_transformers_installed()
            if not installed or input(f"Transformers {version} is already installed. Replace with xLSTM fork? (y/n): ").lower() == 'y':
                install_transformers_fork()
            # Then install xlstm
            self.install_xlstm()
        except Exception as e:
            print(f"Error during development installation: {e}")
            raise

    def install_xlstm(self):
        print("====== Installing AMD-optimized xLSTM library (development mode)... ======")
        try:
            # Get current directory
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
            
            # Create a temporary directory
            temp_dir = os.path.join(current_dir, "xlstm_temp")
            print(f"Temp directory will be: {temp_dir}")
            
            # Remove if it exists
            if os.path.exists(temp_dir):
                print(f"Removing existing temp directory: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove directory: {e}")
                    print("Attempting alternative removal method...")
                    
                    # Try force removal with system commands
                    import time
                    time.sleep(1)  # Wait briefly for any processes to release files
                    
                    try:
                        if os.name == 'nt':  # Windows
                            subprocess.check_call(f'rmdir /S /Q "{temp_dir}"', shell=True)
                        else:  # Linux/Mac
                            subprocess.check_call(f'rm -rf "{temp_dir}"', shell=True)
                    except Exception as e2:
                        print(f"Warning: Alternative removal also failed: {e2}")
                        print("Will attempt to continue with installation anyway")
            
            # Double-check dir is gone before proceeding
            if os.path.exists(temp_dir):
                print(f"Warning: {temp_dir} still exists. Installation may fail.")
                
            print("Cloning xlstm repository...")
            subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", temp_dir])
            
            print("Installing xlstm...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", temp_dir])
            
            print("AMD-optimized xLSTM library installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
            raise
        except Exception as e:
            print(f"Error installing xLSTM: {e}")
            raise

# Alternative post-installation method using a script entry point
def install_xlstm_package():
    """Script entry point for installing xlstm."""
    print("====== Running post-installation script to install AMD-optimized xLSTM library... ======")
    try:
        # Check and install transformers fork
        installed, version = check_transformers_installed()
        if not installed or input(f"Transformers {version} is already installed. Replace with xLSTM fork? (y/n): ").lower() == 'y':
            install_transformers_fork()
            
        # Get current directory
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        
        # Create a temporary directory
        temp_dir = os.path.join(current_dir, "xlstm_temp")
        print(f"Temp directory will be: {temp_dir}")
        
        # Remove if it exists
        if os.path.exists(temp_dir):
            print(f"Removing existing temp directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove directory: {e}")
                print("Attempting alternative removal method...")
                
                # Try force removal with system commands
                import time
                time.sleep(1)  # Wait briefly for any processes to release files
                
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.check_call(f'rmdir /S /Q "{temp_dir}"', shell=True)
                    else:  # Linux/Mac
                        subprocess.check_call(f'rm -rf "{temp_dir}"', shell=True)
                except Exception as e2:
                    print(f"Warning: Alternative removal also failed: {e2}")
                    print("Will attempt to continue with installation anyway")
        
        # Double-check dir is gone before proceeding
        if os.path.exists(temp_dir):
            print(f"Warning: {temp_dir} still exists. Installation may fail.")
            
        print("Cloning xlstm repository...")
        subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", temp_dir])
        
        print("Installing xlstm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", temp_dir])
        
        print("AMD-optimized xLSTM library installed successfully!")
    except Exception as e:
        print(f"Error installing xLSTM: {e}")
        raise

setup(
    name="xlstm-kernels",
    version="2.0.0",
    packages=find_packages(),
    description="AMD-optimized kernels for xLSTM",
    author="Elio",
    author_email="elio.vp@gmail.com",
    python_requires=">=3.11",
    install_requires=[
        "dacite",
        "einops",
        "ipykernel",
        "matplotlib",
        "numpy",
        "omegaconf",
        "rich",
        "torch",
        "tqdm",
        "safetensors",
        "tokenizers>=0.20,<0.21",
        # Note: transformers is handled separately to use the specific fork
    ],
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    entry_points={
        'console_scripts': [
            'install-xlstm=setup:install_xlstm_package',
        ],
    },
    license="Same as xLSTM",
) 