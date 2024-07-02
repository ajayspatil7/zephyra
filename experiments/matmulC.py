import ctypes
import numpy as np
import torch
import time

import pandas as pd

# strassen_in_C
strassen = [0.00040198500028054696, 0.0004494069999054773, 0.00038358600068022497, 0.0003964750012528384, 0.00038412499998230487, 0.0003914259996236069, 0.00039553600072395056, 0.0004185839989077067, 0.0004421379999257624, 0.0004035849997308105]
traditional = [0.0006441400000767317, 0.0006575919996976154, 0.0006631280011788476, 0.0008676840006955899, 0.0003785659991990542, 0.00038142799894558266, 0.0003874079993693158, 0.00038297200080705807, 0.0003835600000456907, 0.00037734299985459074]
# Load the shared library

matrix_lib = ctypes.CDLL('./pipe/matmul.so')
# matrix_lib = ctypes.CDLL('./strassens.so')

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

timer_strassen = []
timer_traditional = []
timer_torch = []


# print('-' * 100)
# print(f"Using a C wrapper function for matrix multiplication using strassens algorithm on a {m}x{m} matrix")
# print('-' * 100)

# st1 = time.perf_counter()
# result = matrix_multiply_python(np.array(matrix1), np.array(matrix2))
# sp1 = time.perf_counter()



m = 25
matrix1 = torch.randint(0, 100, (m, m))
matrix2 = torch.randint(0, 100, (m, m))

st1 = time.perf_counter()
result = matrix_multiply_python(np.array(matrix1), np.array(matrix2))
res = torch.matmul(matrix1, matrix2)
sp1 = time.perf_counter()
timer_torch.append(float(sp1-st1))
# print(f"\nTime took by C wrapper: {sp1 - st1}\n")

print(timer_torch)

# print(result)
# print('-' * 100)

