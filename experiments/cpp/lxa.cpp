#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using Matrix = std::vector<std::vector<int>>;

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



// dummy function to add two matrices
Matrix add_matrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// dummy function to subtract two matrices
Matrix subtract_matrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Strassen's algorithm
Matrix strassen(const Matrix& A, const Matrix& B) {
    int n = A.size();

    // Base case: 1x1 matrix
    if (n == 1) {
        return {{A[0][0] * B[0][0]}};
    }

    // Pad matrices if necessary
    int new_size = n;
    if (n & (n - 1)) {
        new_size = 1;
        while (new_size < n) new_size *= 2;
    }

    Matrix A_padded(new_size, std::vector<int>(new_size, 0));
    Matrix B_padded(new_size, std::vector<int>(new_size, 0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A_padded[i][j] = A[i][j];
            B_padded[i][j] = B[i][j];
        }

    // Divide matrices into quadrants
    int half = new_size / 2;
    Matrix A11(half, std::vector<int>(half)), A12(half, std::vector<int>(half));
    Matrix A21(half, std::vector<int>(half)), A22(half, std::vector<int>(half));
    Matrix B11(half, std::vector<int>(half)), B12(half, std::vector<int>(half));
    Matrix B21(half, std::vector<int>(half)), B22(half, std::vector<int>(half));

    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            A11[i][j] = A_padded[i][j];
            A12[i][j] = A_padded[i][j + half];
            A21[i][j] = A_padded[i + half][j];
            A22[i][j] = A_padded[i + half][j + half];
            B11[i][j] = B_padded[i][j];
            B12[i][j] = B_padded[i][j + half];
            B21[i][j] = B_padded[i + half][j];
            B22[i][j] = B_padded[i + half][j + half];
        }

    // Compute the 7 products recursively
    Matrix P1 = strassen(A11, subtract_matrices(B12, B22));
    Matrix P2 = strassen(add_matrices(A11, A12), B22);
    Matrix P3 = strassen(add_matrices(A21, A22), B11);
    Matrix P4 = strassen(A22, subtract_matrices(B21, B11));
    Matrix P5 = strassen(add_matrices(A11, A22), add_matrices(B11, B22));
    Matrix P6 = strassen(subtract_matrices(A12, A22), add_matrices(B21, B22));
    Matrix P7 = strassen(subtract_matrices(A11, A21), add_matrices(B11, B12));

    // Compute the quadrants of the result
    Matrix C11 = subtract_matrices(add_matrices(add_matrices(P5, P4), P6), P2);
    Matrix C12 = add_matrices(P1, P2);
    Matrix C21 = add_matrices(P3, P4);
    Matrix C22 = subtract_matrices(subtract_matrices(add_matrices(P5, P1), P3), P7);

    // Combine the quadrants into the result matrix
    Matrix C(new_size, std::vector<int>(new_size, 0));
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            C[i][j] = C11[i][j];
            C[i][j + half] = C12[i][j];
            C[i + half][j] = C21[i][j];
            C[i + half][j + half] = C22[i][j];
        }

    // Trim the result matrix if we added padding
    if (new_size != n) {
        Matrix result(n, std::vector<int>(n, 0));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = C[i][j];
        return result;
    }

    return C;
}

PYBIND11_MODULE(lxa, m) {
    m.doc() = "A linear algebra module for python written in C++ currently with matmul() operation only"; // optional module docstring
    m.def("matmul", &matmul, "Traditional matrix multiplication algorithm");
    m.def("strassen", &strassen, "Efficient matrix multiplications algorithm usign Strassen's method");
}