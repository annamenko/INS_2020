from keras import Sequential
from keras.layers import Dense, GRU, Dropout, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from data_manipulation import gen_data_from_sequence


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


def plot_sequence(predicted_res, test_res):
    pred_length = range(len(predicted_res))
    plt.title('Sequence')
    plt.ylabel('Sequence')
    plt.xlabel('x')
    plt.plot(pred_length, predicted_res)
    plt.plot(pred_length, test_res)
    plt.show()


def create_dataset():
    data, res = gen_data_from_sequence()

    dataset_size = len(data)
    train_size = (dataset_size // 10) * 7
    val_size = (dataset_size - train_size) // 2

    train_data, train_res = data[:train_size], res[:train_size]
    val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
    test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]
    return train_data, train_res, val_data, val_res, test_data, test_res


train_data, train_res, val_data, val_res, test_data, test_res = create_dataset()

model = Sequential()
model.add(GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(LSTM(32, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(Dropout(0.5))
model.add(GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(Dense(1))
model.compile(Adam(), loss='mse')
H = model.fit(
        train_data,
        train_res,
        batch_size=5,
        epochs=15,
        verbose=1,
        validation_data=(val_data, val_res)
    )
plot_loss(H.history['loss'], H.history['val_loss'])
plot_sequence(model.predict(test_data), test_res)
