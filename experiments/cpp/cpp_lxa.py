import lxa
import numpy as np

# Matrix dimensions

x, y = 10, 10

A = np.random.randint(100, size=(x, y))
B = np.random.randint(100, size=(x, y))

# Call the C++ matmul function
result = lxa.matmul(A, B)

print("Results:")
for row in result:
    print(row)