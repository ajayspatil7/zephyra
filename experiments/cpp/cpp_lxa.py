import lxa
import numpy as np
import time
import sys

# Matrix dimensions
# Currently available functions in lxa module: strassen(mat1, mat2), matmul(mat1, mat2)


def timeIT(dimX: int, dimY: int, runs: int, method: str) -> float:

    """
    Calculating runtime for the algorithm wrappers
    :dimX : rows 
    :dimY : cols
    :runs : no of times to perform the experiment
    :method : which method to use
    """

    counter = []

    if method.isnumeric():
        raise TypeError(f"TypeError: timeIT() does not support {type(method)} supports only type{str}")
    
    else:
        if method.lower() == "trad":
            for x in range(runs):
                A = np.random.randint(100, size=(dimX, dimY))
                B = np.random.randint(100, size=(dimX, dimX))
                st = time.perf_counter()
                result = lxa.matmul(A, B)
                sp = time.perf_counter()
                counter.append(sp - st)

        if method.lower() == "stra":
            for x in range(runs):
                A = np.random.randint(100, size=(dimX, dimY))
                B = np.random.randint(100, size=(dimX, dimX))
                st = time.perf_counter()
                result = lxa.strassen(A, B)
                sp = time.perf_counter()
                counter.append(sp - st)
    
    avg = sum(counter) / runs
    return avg

x, y = int(sys.argv[1]), int(sys.argv[2])
runs = int(sys.argv[3])
method = sys.argv[4]

avg = timeIT(x, y, 100, method)
print(f"Average time took for {method} as C++ wrapper for a {x}x{y} matrix: {avg:.5f} seconds\n")
