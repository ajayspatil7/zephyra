#include <stdlib.h>
#include <string.h>

void add_matrices(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] + B[i];
    }
}

void subtract_matrices(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] - B[i];
    }
}

void strassen_multiply(int* A, int* B, int* C, int size) {
    if (size == 1) {
        C[0] = A[0] * B[0];
        return;
    }

    int newSize = size / 2;
    int* A11 = (int*)malloc(newSize * newSize * sizeof(int));
    int* A12 = (int*)malloc(newSize * newSize * sizeof(int));
    int* A21 = (int*)malloc(newSize * newSize * sizeof(int));
    int* A22 = (int*)malloc(newSize * newSize * sizeof(int));
    int* B11 = (int*)malloc(newSize * newSize * sizeof(int));
    int* B12 = (int*)malloc(newSize * newSize * sizeof(int));
    int* B21 = (int*)malloc(newSize * newSize * sizeof(int));
    int* B22 = (int*)malloc(newSize * newSize * sizeof(int));
    int* C11 = (int*)malloc(newSize * newSize * sizeof(int));
    int* C12 = (int*)malloc(newSize * newSize * sizeof(int));
    int* C21 = (int*)malloc(newSize * newSize * sizeof(int));
    int* C22 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M1 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M2 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M3 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M4 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M5 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M6 = (int*)malloc(newSize * newSize * sizeof(int));
    int* M7 = (int*)malloc(newSize * newSize * sizeof(int));
    int* temp1 = (int*)malloc(newSize * newSize * sizeof(int));
    int* temp2 = (int*)malloc(newSize * newSize * sizeof(int));

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            int idx1 = i * newSize + j;
            int idx2 = i * size + j;
            int idx3 = i * size + j + newSize;
            int idx4 = (i + newSize) * size + j;
            int idx5 = (i + newSize) * size + j + newSize;
            
            A11[idx1] = A[idx2];
            A12[idx1] = A[idx3];
            A21[idx1] = A[idx4];
            A22[idx1] = A[idx5];
            
            B11[idx1] = B[idx2];
            B12[idx1] = B[idx3];
            B21[idx1] = B[idx4];
            B22[idx1] = B[idx5];
        }
    }

    add_matrices(A11, A22, temp1, newSize);
    add_matrices(B11, B22, temp2, newSize);
    strassen_multiply(temp1, temp2, M1, newSize);

    add_matrices(A21, A22, temp1, newSize);
    strassen_multiply(temp1, B11, M2, newSize);

    subtract_matrices(B12, B22, temp1, newSize);
    strassen_multiply(A11, temp1, M3, newSize);

    subtract_matrices(B21, B11, temp1, newSize);
    strassen_multiply(A22, temp1, M4, newSize);

    add_matrices(A11, A12, temp1, newSize);
    strassen_multiply(temp1, B22, M5, newSize);

    subtract_matrices(A21, A11, temp1, newSize);
    add_matrices(B11, B12, temp2, newSize);
    strassen_multiply(temp1, temp2, M6, newSize);

    subtract_matrices(A12, A22, temp1, newSize);
    add_matrices(B21, B22, temp2, newSize);
    strassen_multiply(temp1, temp2, M7, newSize);

    add_matrices(M1, M4, temp1, newSize);
    subtract_matrices(temp1, M5, temp2, newSize);
    add_matrices(temp2, M7, C11, newSize);

    add_matrices(M3, M5, C12, newSize);
    add_matrices(M2, M4, C21, newSize);

    add_matrices(M1, M3, temp1, newSize);
    subtract_matrices(temp1, M2, temp2, newSize);
    add_matrices(temp2, M6, C22, newSize);

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            int idx1 = i * newSize + j;
            int idx2 = i * size + j;
            int idx3 = i * size + j + newSize;
            int idx4 = (i + newSize) * size + j;
            int idx5 = (i + newSize) * size + j + newSize;
            
            C[idx2] = C11[idx1];
            C[idx3] = C12[idx1];
            C[idx4] = C21[idx1];
            C[idx5] = C22[idx1];
        }
    }

    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(C11);
    free(C12);
    free(C21);
    free(C22);
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
    free(temp1);
    free(temp2);
}

void matrix_multiply(int* matrix1, int* matrix2, int* result, int rows1, int cols1, int cols2) {
    if (rows1 != cols1 || cols1 != cols2 || (rows1 & (rows1 - 1)) != 0) {
        // Fallback to the standard algorithm if matrices are not 2^n x 2^n
        for (int i = 0; i < rows1 * cols2; i++) {
            result[i] = 0;
        }
        
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
                }
            }
        }
    } else {
        strassen_multiply(matrix1, matrix2, result, rows1);
    }
}
