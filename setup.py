from setuptools import setup, find_packages
import subprocess
import os
import sys
from setuptools.command.install import install
from setuptools.command.develop import develop

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # First run the standard install
        install.run(self)
        # Then install xlstm
        self.install_xlstm()

    def install_xlstm(self):
        print("Installing AMD-optimized xLSTM library...")
        # Create a temporary directory
        subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", "xlstm_temp"])
        # Install xlstm
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "./xlstm_temp"])
        print("AMD-optimized xLSTM library installed successfully!")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # First run the standard develop
        develop.run(self)
        # Then install xlstm
        self.install_xlstm()

    def install_xlstm(self):
        print("Installing AMD-optimized xLSTM library...")
        # Create a temporary directory
        if not os.path.exists("xlstm_temp"):
            subprocess.check_call(["git", "clone", "https://github.com/Eliovp/xlstm.git", "xlstm_temp"])
        # Install xlstm
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "./xlstm_temp"])
        print("AMD-optimized xLSTM library installed successfully!")

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
        "transformers",
        "safetensors",
    ],
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    license="Same as xLSTM",
) 