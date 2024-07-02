#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }

    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

PYBIND11_MODULE(lxa, m) {
    m.doc() = "A linear algebra module for python written in C++"; // optional module docstring
    m.def("matmul", &matmul, "A function that multiplies two matrices");
}