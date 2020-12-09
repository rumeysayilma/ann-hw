import matplotlib.pyplot as plt
import numpy as np
image = np.ones((5, 10))
image[:, 0] = 0
image[:, -1] = 0
image[0, :] = 0
image[-1, :] = 0
print(image[0])
plt.imshow(image, cmap='gray')
plt.show()
noisy = image + 0.2 * np.random.rand(5, 10)
noisy = noisy/noisy.max()
plt.imshow(noisy, cmap='gray')
plt.show()
rand_index = np.random.randint(0, 50)
rand_image = np.array(image)
rand_image[int(rand_index % 5),
 int(rand_index / 5)] = np.where(
 rand_image[int(rand_index % 5), int(rand_index / 5)] == 1,
 0.0, 1.0)
plt.imshow(rand_image, cmap='gray')
plt.show()