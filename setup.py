from setuptools import find_packages, setup

setup(
    name="flowadam",
    version="0.1.0",
    description="FlowAdam optimizer with geometry-aware soft momentum injection",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchdiffeq>=0.2",
    ],
)
