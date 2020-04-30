import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from callback import MyCallBack as CB
from var2 import gen_data

NM = 2

import matplotlib.pyplot as plt


def plot_loss(loss, v_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(loss, 'b', label='train')
    plt.plot(v_loss, 'r', label='validation')
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def plot_acc(acc, val_acc):
    plt.plot(acc, 'b', label='train')
    plt.plot(val_acc, 'r', label='validation')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def generation_data():
    x, y = gen_data(size=1000, img_size=28)
    x, y = np.asarray(x), np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train, NM)

    encoder.fit(y_test)
    y_test = encoder.transform(y_test)
    y_test = to_categorical(y_test, NM)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = generation_data()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=(28, 28, 1), name='first'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='mp2d_3'))
model.add(Dropout(0.25, name='first_dropout'))
model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu', name='simple_dense'))
model.add(Dropout(0.5, name='second_dropout'))
model.add(Dense(NM, activation='softmax', name='last_layer'))

model.compile(Adam(lr=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
H = model.fit(
            x_train,
            y_train,
            batch_size=100,
            epochs=80,
            verbose=1,
            shuffle=True,
            validation_data=(x_test, y_test),
            validation_split=0.1,
            callbacks=[CB([1])]
        )
_, acc = model.evaluate(x_train, y_train)
print('Train', acc)
_, acc = model.evaluate(x_test, y_test)
print('Test', acc)

plot_loss(H.history['loss'], H.history['val_loss'])
plot_acc(H.history['accuracy'], H.history['val_accuracy'])


