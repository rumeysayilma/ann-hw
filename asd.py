import numpy as np
a = np.array([0.67, 0, 0.7])
b = np.where(a == np.max(a), 1, 0)
print(np.equal(np.zeros((3, 1)), np.zeros((3, 1))))
''' np.where(a == np.max(a), 1, 0) '''

b = np.where(a == np.max(a), 1, 0)
print(b)
