from setuptools import setup, find_packages

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
        # Install xLSTM from GitHub - will need to be updated once the AMD-optimized xLSTM repo is created
        "xlstm @ git+https://github.com/Eliovp/xlstm.git",
    ],
    license="Same as xLSTM",
) 