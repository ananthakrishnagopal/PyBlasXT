from src import multiply

import numpy as np

a = np.random.rand(1000,5).astype(np.float32)
b = np.random.rand(5,1000).astype(np.float32)
print(np.linalg.norm(multiply.multiply(a,b,np.array([0]))-a@b))
