#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

// Create a Python module named 'example'
PYBIND11_MODULE(matmulcpp, m) {
    
    m.def("add", &add, "A function that adds two numbers");
}
