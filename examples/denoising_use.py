import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

def add_noise(noise_factor, img):
    return np.random.normal(0, noise_factor, 28 * 28).reshape(28, 28) + img

noise_factor = 0.5

_, (x_test, _) = mnist.load_data()
x_test = x_test / 255.0
x_test_noisy = add_noise(noise_factor, x_test)

autoencoder = load_model("models/denoising.keras")

predicted = autoencoder.predict(x_test_noisy)

# Plotagem

nrows = 2
ncols = 5

for i in range(1, ncols + 1):
    plt.subplot(nrows, ncols, i)
    plt.axis("off")
    plt.imshow(x_test_noisy[i], cmap="gray")
    plt.subplot(nrows, ncols, i + ncols)
    plt.axis("off")
    plt.imshow(predicted[i], cmap="gray")

plt.show()
