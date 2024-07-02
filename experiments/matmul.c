// matrix_multiply.c
#include <stdlib.h>

void matrix_multiply(int* matrix1, int* matrix2, int* result, int rows1, int cols1, int cols2) {
    // Initialize result matrix to 0
    for (int i = 0; i < rows1 * cols2; i++) {
        result[i] = 0;
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
            }
        }
    }
}
