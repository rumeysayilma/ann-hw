import numpy as np
import matplotlib.pyplot as plt
import cv2

x= np.zeros((50,50))
""" y=np.reshape(x, (50,1)) """

""" print(np.random.rand(50, 50)) """
noisy = x + 0.1 * np.random.rand(50, 50)

noisy = noisy/noisy.max()

plt.imshow(noisy, cmap='gray')
plt.show()

cv2.imshow('s',noisy)
cv2.waitKey(0)