import numpy as np
import matplotlib.pyplot as plt
import cv2

size=[50,4,4]

weights = []
for i in range(1, len(size)):
    weights.append(np.random.rand(size[i], size[i-1]))



gradients =[]

for i in range(len(e)):
    gradients.append([])
    for l  in range(len(size)-1):
        gradients[i].append([])
      

"""  """

for i  in range(len(e)):
    gradients[i][-1] = gradyen_output[i][0]
    print(gradients[i][-1])

