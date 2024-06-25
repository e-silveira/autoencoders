import matplotlib.pyplot as plt
import cv2 as cv


img = plt.imread("../img/turing.jpg")
img = img / 255.0
img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)

plt.imshow(img, cmap="gray")
plt.show()
