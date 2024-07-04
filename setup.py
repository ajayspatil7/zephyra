from setuptools import setup, find_packages

setup(
    name="zephyra",
    version="0.1",
    packages=find_packages(),
    package_data={'': ['*.so', '*.pyd']},  # Include compiled extensions
)