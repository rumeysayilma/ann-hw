import matplotlib.pyplot as plt
import numpy as np

#harflerin ve 4 bit hatası yapılmış versiyonlarının oluşturulması
def pattern_designer():
    t= np.ones((5, 10),dtype=float)
    t [:, 5] = 0
    t[0, :] = 0
    t_degisik = np.copy(t)
    t_degisik[0 ,0],t_degisik[1 ,0],t_degisik[3 ,4],t_degisik[3 ,5] = 1,0,0,1
    l= np.ones((5, 10),dtype=float)
    l[:, 0] = 0
    l[-1, :] = 0
    l_degisik = np.copy(l)
    l_degisik[4 ,4],l_degisik[3, 4],l_degisik[4, 9],l_degisik[3 ,9] = 1,0,1,0
    h= np.ones((5, 10),dtype=float)
    h[:, 2] = 0
    h[:, 6] = 0
    h[2, :] = 0
    h_degisik = np.copy(h)
    h_degisik[0,8],h_degisik[2,6],h_degisik[0,0],h_degisik[2,2] = 0,1,0,1
    a= np.ones((5, 10),dtype=float)
    a[:, 9] = 0
    a[0, :] = 0
    a_degisik = np.copy(a)
    a_degisik[0, 2],a_degisik[1, 2],a_degisik[3, 8],a_degisik[3, 9] = 1,0,0,1
    return t, h, a, l, t_degisik, h_degisik, a_degisik, l_degisik 

#harfe gürültü ekleme
def gray_noise(harf):
    new_signal = harf + np.random.normal(0, .01, harf.shape)
    return new_signal

t, h, a, l, t_degisik, h_degisik, a_degisik, l_degisik  = pattern_designer()
""" print(t, h, a, l)
 """
print(type(t))
""" image=gray_noise(image) """

""" noisy= image + 0.2* np.random.rand(5,10) """
""" print(image[0]) """
for i in [t,h,l,a,t_degisik, h_degisik, a_degisik, l_degisik ]:
    plt.imshow(i, cmap='gray')
    plt.show()

#0 ve 1 in 0.1 ve 0.9 a çekilmesi için kullanılacak
print(a)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(np.where(a < 4, -1, 100))
# [[ -1  -1  -1]
#  [ -1 100 100]
#  [100 100 100]] """
#önce total veri kümesini yd ler ile beraber birleştir. 
#farklı veiler ile test kümesi oluşturulması
#Tüm verileri vektöre çevir
#gradyan hesaplayan fonkfiyon, aktivasyon fonksiyon, momentum terimi unutma
#ileri yol ve geri yol fonkfiyon
#ağırlık göncelleme
#hata hesap-grafik