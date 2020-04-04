from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.models import Model, Sequential
import numpy as np
import csv
import matplotlib.pyplot as plt


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        try:
            for i in data:
                my_csv.writerow(i)
        except Exception as ex:
            my_csv.writerow(data)


def generation_of_dataset(size_of_dataset):
    dataset = []
    dataset_y = []
    for i in range(size_of_dataset):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        x_sample = []
        x_sample.append(np.round(X ** 2 + e, decimals=4))
        x_sample.append(np.round(np.cos(2*X) + e, decimals=4))
        x_sample.append(np.round(X - 3 + e, decimals=4))
        x_sample.append(np.round(-X + e, decimals=4))
        x_sample.append(np.round(np.fabs(X)+ e, decimals=4))
        x_sample.append(np.round((X**3)/4 + e, decimals=4))
        dataset.append(x_sample)
        dataset_y.append(np.round(np.sin(X/2) + e))
    return np.round(np.array(dataset), decimals=3), np.array(dataset_y)


def build_model():
    model = Sequential()
    model.add(Dense(60, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_objects():
    main_input = Input(shape=(6,), name='main_input')
    encoded = Dense(60, activation='relu')(main_input)
    encoded = Dense(60, activation='relu')(encoded)
    encoded = Dense(45, activation='relu')(encoded)
    encoded = Dense(3, activation='linear')(encoded)

    input_encoded = Input(shape=(3,), name='input_encoded')
    decoded = Dense(35, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(6, name="out_aux")(decoded)

    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")
    return encoded, decoded, predicted


x_train, y_train = generation_of_dataset(300)
x_test, y_test = generation_of_dataset(60)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

encoded, decoded, my_model = create_objects()

keras_model = build_model()
H = keras_model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=2,
                    verbose=1,
                    validation_data=(x_test, y_test))

loss = H.history['loss']
v_loss = H.history['val_loss']

plt.plot(range(1, 51), loss, 'b', label='train')
plt.plot(range(1, 51), v_loss, 'r', label='validation')
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

my_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
H = my_model.fit(x_train, y_train,
                 epochs=50,
                 batch_size=2,
                 verbose=1,
                 validation_data=(x_test, y_test))

encoded_data = encoded.predict(x_test)
decoded_data = decoded.predict(encoded_data)

loss = H.history['loss']
v_loss = H.history['val_loss']
x = range(1, 51)

plt.plot(x, loss, 'b', label='train')
plt.plot(x, v_loss, 'r', label='validation')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

write_csv('./x_train.csv', x_train)
write_csv('./y_train.csv', y_train)
write_csv('./x_test.csv', x_test)
write_csv('./y_test.csv', y_test)
write_csv('./encoded.csv', encoded_data)
write_csv('./decoded.csv', decoded_data)

# save models
decoded.save('decoder.h5')
encoded.save('encoder.h5')
my_model.save('full.h5')
