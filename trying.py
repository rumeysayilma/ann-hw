import numpy as np
import matplotlib.pyplot as plt
import cv2

t = np.ones((5, 10))
t[:, 5] = 0
t[0, :] = 0
t = np.where(t == 0, 0.1, 0.9)
t_degisik = np.copy(t)
t_degisik[0, 0], t_degisik[1, 0], t_degisik[3, 4], t_degisik[3, 5] = 1, 0, 0, 1
print(t_degisik)
for i in [t, t_degisik]:
    plt.imshow(i, cmap='gray')
    plt.show()

""" print(t)
x_tryi = [t,l]
x_tryi = [a.reshape((50,1)) for a in x_tryi] """

""" t = np.reshape(np.where(t == 0, 0.1, 0.9), (50, 1)) """
print(t)
print(x_tryi)
print(l)
