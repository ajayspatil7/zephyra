# setup.py
from setuptools import setup, find_packages

setup(
    name="zephyra",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "tiktoken>=0.3.0",
        "tqdm>=4.62.0",
    ],
)
