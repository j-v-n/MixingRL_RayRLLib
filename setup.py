from setuptools import setup

setup(
    name="mixing_environment",
    version="1.0.0",
    description="environment for simulating mixing problem",
    author="Jayanth Nair",
    python_requires="~=3.9",
    install_requires=[
        "gymnasium",
        "ray[rllib]",
        "tensorflow",
        "scipy",
        "matplotlib",
        "numpy",
    ],
)
