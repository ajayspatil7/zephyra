import lxa
import numpy as np

# Example matrices
A = np.random.randint(10, size=(5, 5))
B = np.random.randint(10, size=(5, 5))

# Call the C++ matmul function
result = lxa.matmul(A, B)

print("Result of matrix multiplication:")
for row in result:
    print(row)