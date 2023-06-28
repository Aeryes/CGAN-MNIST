from tensorflow.keras.layers import UpSampling2D, Reshape, Activation, Conv2D, BatchNormalization, LeakyReLU, Input, Flatten, multiply
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import warnings

import os

warnings.filterwarnings('ignore')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def build_generator(z_dim, num_classes):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(z_dim,)))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(1, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('tanh'))

    z = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)
    label_embedding = Flatten()(label_embedding)
    joined = multiply([z, label_embedding])

    img = model(joined)
    return Model([z, label], img)

def build_discriminator(img_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=(img_shape))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(input_dim=num_classes, output_dim=np.prod(img_shape), input_length=1)(label)
    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape(img_shape)(label_embedding)

    concat = Concatenate(axis=-1)([img, label_embedding])
    prediction = model(concat)
    return Model([img, label], prediction)

def train(generator, discriminator, epochs, batch_size, save_interval, z_dim, num_classes, cgan):
    (X_train, y_train), (_, _) = mnist.load_data()

    X_train = (X_train - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    real = np.ones(shape=(batch_size, 1))
    fake = np.zeros(shape=(batch_size, 1))

    for iteration in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        gen_imgs = generator.predict([z, labels])

        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        g_loss = cgan.train_on_batch([z, labels], real)

        if iteration % save_interval == 0:
            print(
                '{} [D loss: {}, accuracy: {:.2f}] [G loss: {}]'.format(iteration, d_loss[0], 100 * d_loss[1], g_loss))
            save_image(generator, iteration, z_dim)

def save_image(generator, epoch, z_dim):
    r, c = 2, 5
    z = np.random.normal(0, 1, (r * c, z_dim))
    labels = np.arange(0, 10).reshape(-1, 1)
    gen_image = generator.predict([z, labels])
    gen_image = 0.5 * gen_image + 0.5

    fig, axes = plt.subplots(r, c, figsize=(10, 10))
    count = 0
    for i in range(r):
        for j in range(c):
            axes[i, j].imshow(gen_image[count, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title("Digit: %d" % labels[count])
            count += 1
    plt.savefig('images/cgan_%d.jpg' % epoch)
    plt.close()

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    img_width, img_height = 28, 28
    img_channel = 1
    img_shape = (img_width, img_height, img_channel)
    num_classes = 10
    z_dim = 100

    generator = build_generator(z_dim, num_classes)
    generator.summary()

    discriminator = build_discriminator(img_shape, num_classes)
    discriminator.summary()

    X_train.shape

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5), metrics=['accuracy'])

    z = Input(shape=(z_dim,))
    label = Input(shape=(1,))
    img = generator([z, label])

    # discriminator.trainable = False
    prediction = discriminator([img, label])

    cgan = Model([z, label], prediction)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5))

    os.mkdir("images/")

    # training the network
    train(generator, discriminator, 10000, 128, 1000, z_dim, num_classes, cgan)


if __name__ == '__main__':
    main()


