from setuptools import setup, Extension
import pybind11
import sys

# Specify the compiler flags
extra_compile_args = ['-std=c++11', '-O3']

# For macOS, we need to specify the minimum version
if sys.platform == 'darwin':
    extra_compile_args.append('-mmacosx-version-min=10.9')

ext_modules = [
    Extension(
        'lxa',  # name of the module
        ['lxa.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='lxa',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    setup_requires=['pybind11>=2.5.0'],
)