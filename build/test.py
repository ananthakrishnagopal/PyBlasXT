from src import PyBlasXT

import numpy as np
import time
a = np.random.rand(1000,5000)
b = np.random.rand(5000,1000)
t0 = time.time()
PyBlasXT.dgemm(a,b,np.array([0]))
t1 = time.time()
print(t1-t0)
t0 = time.time()
a@b
t1 = time.time()
print(t1-t0)

print(np.linalg.norm(PyBlasXT.dgemm(a,b,np.array([0]))-a@b))
