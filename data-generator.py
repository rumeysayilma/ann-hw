import matplotlib.pyplot as plt
import numpy as np

def pattern_designer():
    t= np.ones((5, 10),dtype=float)
    t [:, 5] = 0
    t[0, :] = 0
    t_degisik = t
    t_degisik[0 ,0],t_degisik[1 ,0],t_degisik[3 ,4],t_degisik[3 ,5] = 1,0,0,1
    l= np.ones((5, 10),dtype=float)
    l[:, 0] = 0
    l[-1, :] = 0
    l_degisik = l
    l_degisik[4 ,4],l_degisik[3, 4],l_degisik[4, 9],l_degisik[3 ,9] = 1,0,1,0
    h= np.ones((5, 10),dtype=float)
    h[:, 2] = 0
    h[:, 6] = 0
    h[2, :] = 0
    h_degisik = h
    h_degisik[0,8],h_degisik[2,6],h_degisik[0,0],h_degisik[2,2] = 0,1,0,1
    a= np.ones((5, 10),dtype=float)
    a[:, 9] = 0
    a[0, :] = 0
    a_degisik = a
    a_degisik[0, 2],a_degisik[1, 2],a_degisik[3, 8],a_degisik[3, 9] = 1,0,0,1
    return t_degisik, h_degisik, a_degisik, l_degisik 

def gray_noise(harf):
    new_signal = harf + np.random.normal(0, .01, harf.shape)
    return new_signal

t, h, a, l = pattern_designer()
""" print(t, h, a, l)
 """
print(type(t))
""" image=gray_noise(image) """

""" noisy= image + 0.2* np.random.rand(5,10) """
""" print(image[0]) """
for i in [t,h,l,a]:
    plt.imshow(i, cmap='gray')
    plt.show()


print(a)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(np.where(a < 4, -1, 100))
# [[ -1  -1  -1]
#  [ -1 100 100]
#  [100 100 100]] """