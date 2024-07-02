cmake_minimum_required(VERSION 3.0)
project(matmulcpp)

# Set the C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)

# Find and include the pybind11 package
find_package(pybind11 REQUIRED)

# Add the module to be built
pybind11_add_module(matmulcpp matmul.cpp)
