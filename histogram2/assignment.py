import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")

B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

cv2.imshow("Wheat Field with Cypresses", img)
cv2.imshow("Blue", B)
cv2.imshow("Green", G)
cv2.imshow("Red", R)
cv2.waitKey()

imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGray, cmap='gray')
plt.show

h = np.zeros(256, np.int32)
print(h)

row, col = img.shape[0], img.shape[1]
print(img.shape)

for i in range(0, row):
    for j in range(0, col):
        h[img[i, j]] += 1

plt.figure()
plt.plot(h)
plt.show()
