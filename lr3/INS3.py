import tensorflow
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


class lab3:
    def __init__(self, epochs_count=100):
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = boston_housing.load_data()
        self.mean = self.train_data.mean(axis=0)
        self.train_data -= self.mean
        self.std = self.train_data.std(axis=0)
        self.train_data /= self.std
        self.model = self.build_model()
        self.test_data -= self.mean
        self.test_data /= self.std
        self.k = 10
        self.num_val_samples = len(self.train_data) // self.k
        self.num_epochs = epochs_count
        self.all_scores = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.train_data.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model

    def find_overfit(self):
        for i in range(3, self.k):
            print('processing fold #', i)
            val_data = self.train_data[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            val_targets = self.train_targets[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            partial_train_data = np.concatenate(
                [self.train_data[:i * self.num_val_samples], self.train_data[(i + 1) * self.num_val_samples:]], axis=0)
            partial_train_targets = np.concatenate(
                [self.train_targets[:i * self.num_val_samples], self.train_targets[(i + 1) * self.num_val_samples:]],
                axis=0)
            self.model = self.build_model()
            history = self.model.fit(partial_train_data, partial_train_targets, epochs=self.num_epochs, batch_size=1,
                                     verbose=0, validation_data=(val_data, val_targets))
            loss = history.history['loss']
            mae = history.history['mean_absolute_error']
            v_loss = history.history['val_loss']
            v_mae = history.history['val_mean_absolute_error']
            x = range(1, self.num_epochs + 1)

            val_mse, val_mae = self.model.evaluate(val_data, val_targets, verbose=0)
            self.all_scores.append(val_mae)
            plt.plot(x, loss)
            plt.plot(x, v_loss)
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend(['Train data', 'Test data'], loc='upper left')
            plt.show()

            plt.plot(x, mae)
            plt.plot(x, v_mae)
            plt.title('Model mean absolute error')
            plt.ylabel('mean absolute error')
            plt.xlabel('epochs')
            plt.legend(['Train data', 'Test data'], loc='upper left')
            plt.show()

    def fit_model(self):
        res = []
        for i in range(self.k):
            print('processing fold #', i)
            val_data = self.train_data[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            val_targets = self.train_targets[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            partial_train_data = np.concatenate([self.train_data[:i * self.num_val_samples], self.train_data[(i + 1) * self.num_val_samples:]], axis=0)
            partial_train_targets = np.concatenate(
                [self.train_targets[:i * self.num_val_samples], self.train_targets[(i + 1) * self.num_val_samples:]],
                axis=0)
            self.model = self.build_model()
            history = self.model.fit(partial_train_data, partial_train_targets, epochs=self.num_epochs, batch_size=1,
                                verbose=0)
            val_mse, val_mae = self.model.evaluate(val_data, val_targets, verbose=0)
            self.all_scores.append(val_mae)
            res.append(np.mean(self.all_scores))
        plt.plot(range(self.k), res)
        plt.title('Dependence on k')
        plt.ylabel('Mean')
        plt.xlabel('k')
        plt.show()
        print(np.mean(self.all_scores))



lab = lab3(30)
lab.fit_model()

