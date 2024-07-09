from keras.datasets import mnist
from keras.layers import Dense, Flatten, GaussianNoise, Reshape
from keras.models import Sequential

(x_train, _), _ = mnist.load_data()
x_train = x_train / 255.0

noise_factor = 0.5

autoencoder = Sequential([
    Flatten(),
    GaussianNoise(stddev=noise_factor), # Usada somente durante o treinamento.
    Dense(128, activation='relu'),
    Dense(28 * 28, activation='sigmoid'), # Sigmoid para normalizar os dados.
    Reshape([28, 28])
])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256)

autoencoder.save("models/denoising.keras")
