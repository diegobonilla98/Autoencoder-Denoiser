from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# add noise
x_train = x_train + np.random.rand(*x_train.shape)
x_test = x_test + np.random.rand(*x_test.shape)

input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
latent_dim = 16
layer_filters = [32, 64]

# encoder
inputs = Input(shape=input_shape, name="encoder_input")
x = inputs
for filters in layer_filters:
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

# guess the shape of the output
shape = K.int_shape(x)

x = Flatten()(x)
latent = Dense(latent_dim, name="latent_vector")(x)

encoder = Model(inputs, latent, name="encoder")
encoder.summary()
# plot_model(encoder, to_file="encoder.png", show_shapes=True)


# decoder
latent_inputs = Input(shape=(latent_dim, ), name="decoder_input")
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size,
                          activation='sigmoid', padding='same',
                          name="decoder_output")(x)

decoder = Model(latent_inputs, outputs, name="decoder")
decoder.summary()
# plot_model(decoder, to_file='decoder.png', show_shapes=True)


# autoencoder = encoder + decoder
autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
autoencoder.summary()
plot_model(autoencoder, to_file="autoencoder.png", show_shapes=True)

autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=1, batch_size=batch_size)

x_decoded = autoencoder.predict(x_test)

imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])

plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('noisy_input_and_decoded.png')
plt.show()

