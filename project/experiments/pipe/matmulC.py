import ctypes
import numpy as np
import torch
import time

import pandas as pd

# strassen_in_C
# Load the shared library

# matrix_lib = ctypes.CDLL('./matmul.so')
matrix_lib = ctypes.CDLL('./strassens.so')

# Define the argument and return types
matrix_lib.matrix_multiply.argtypes = [
    ctypes.POINTER(ctypes.c_int), # matrix1
    ctypes.POINTER(ctypes.c_int), # matrix2
    ctypes.POINTER(ctypes.c_int), # result
    ctypes.c_int,                 # rows1
    ctypes.c_int,                 # cols1
    ctypes.c_int                  # cols2
]
matrix_lib.matrix_multiply.restype = None


def matrix_multiply_python(matrix1, matrix2):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    
    if cols1 != rows2:
        raise ValueError("Number of columns in matrix1 must be equal to number of rows in matrix2")

    # Prepare the result matrix
    result = np.zeros((rows1, cols2), dtype=np.int32)

    # Convert numpy arrays to ctypes pointers
    matrix1_ptr = matrix1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    matrix2_ptr = matrix2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Call the C function
    matrix_lib.matrix_multiply(matrix1_ptr, matrix2_ptr, result_ptr, rows1, cols1, cols2)

    return result

# Example usage
# matrix1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
# matrix2 = np.array([[70, 80], [90, 10], [11, 12]], dtype=np.int32)

# m = 5


# matrix1 = torch.randint(0, 100, (m, m))
# matrix2 = torch.randint(0, 100, (m, m))

# print("Matrix 1:")
# print(matrix1)

# print("Matrix 2:")
# print(matrix2)

# print('-' * 100)
# print(f"Using Pytorch's torch.matmul() for performing matrix multiplication on ({m}x{n}) matrix")
# print('-' * 100)
# st = time.perf_counter()
# res = torch.matmul(matrix1, matrix2)
# sp = time.perf_counter()

# print(f"\nTime took by torch.matmul() for matrix multiplication: {sp - st}")
# print("Result:")
# print(res)
# print('-' * 100)

timer_strassen = [2.0]
timer_traditional = [2.97]
timer_torch = [1.98]


# print('-' * 100)
# print(f"Using a C wrapper function for matrix multiplication using strassens algorithm on a {m}x{m} matrix")
# print('-' * 100)

# st1 = time.perf_counter()
# result = matrix_multiply_python(np.array(matrix1), np.array(matrix2))
# sp1 = time.perf_counter()



m = 25
matrix1 = torch.randint(0, 100, (m, m))
matrix2 = torch.randint(0, 100, (m, m))

st1 = time.time_ns()
# result = matrix_multiply_python(np.array(matrix1), np.array(matrix2))
res = torch.matmul(matrix1, matrix2)
sp1 = time.time_ns()
# print(f"\nTime took by C wrapper: {sp1 - st1}\n")

print((sp1 - st1) / 1e-9)

# print(result)
# print('-' * 100)

