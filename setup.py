from setuptools import setup, find_packages

setup(
    name="zephyra",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "zephyra=zephyra.run:main",
        ],
    },
)