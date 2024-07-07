"""
Motivation          :       To generate a set of orthonormal vectors from a set of linearly independent vectors
Orthonormal vectors :       Set of unit vectors such that dot product between two vectors is zero

Algorithm,
    --Considering a vector 'A' with 3 linearly independent vectors A(N1), A2(N2), A3(N3)--
    --Let's take an example with A1, A2, A3 as linearly independent vectors--
        
        Step 1 : construct a unit vector along A1 and denote this unit vector as 'e1'
            e1 = (A1 / |A1| ) where |A1| = sqrt(a^2 + b^2) a and b are the values of the vector A1
        
        Step 2 : generate a vector e2 from A2, which is perpendicular to 'e1' using the formula,
            e2 = A2 - (A2 * e1) * e1

        Step 3 : generate a unit vector along e2
            e2 = (e2 / |e2|) where |e2| = sqrt(a^2 + b^2) a and b are the values of the vector e2
        
        Step 4 : generate a vector from A3 which is perpendicular to both e1 and e2 and denote it as e3
            e3 = A3 - (A3 * e1) * e1 - (A3 * e2) * e2
        
        Step 5 : construct unit vector along e3
            e3 = (e3 / |e3|) where |e3| = sqrt(a^2 + b^2) a and b are the values of the vector e3
    
    We can test it out using the following vectors
    A1 = [1, 1, 0]
    A2 = [1, 2, 0]
    A3 = [0, 1, 2]
"""
import numpy as np
import math


def gramSchmidt(v1: list[int], v2: list[int], v3: list[int]) -> list:
    
    # step 1  : Construct unit vector
    
    # step 1a : calculate the magnitude of given vectors
    magnitude_v1 = math.sqrt(sum(v1))
    magnitude_v2 = math.sqrt(sum(v2))
    magnitude_v3 = math.sqrt(sum(v3))


    # step 1b : construct unit vectors for all the vectors
    e1 = [(1/magnitude_v1) * v1[i] for i in range(len(v1))]
    e2 = [(1/magnitude_v2) * v2[i] for i in range(len(v2))]
    e3 = [(1/magnitude_v3) * v3[i] for i in range(len(v3))]

    # step 2  : Generate a vector e2, which is perpendicular to 'e1' using the formula, e2 = A2 - (A2 * e1) * e1

    # step 2a : calculate the dot product of a2 and e1 here a2 ~ v2  -- sum(v2[i] * e2[i])
    a2_dot_e1 = sum([i*j for (i, j) in zip(v2, e1)])
    a2_prod_e1 = [(a2_dot_e1 * e1[i]) for i in range(len(e1))]
    prependicular = a2_dot_e1 - v2

    
    # step 3 : Generate a vector from A3 which is perpendicular to both e1 and e2 and denote it as e3

    

    

v1 = [1, 1, 0]
v2 = [1, 2, 0]
v3 = [0, 1, 2]

gs = gramSchmidt(v1, v2, v3)
# print(gs)