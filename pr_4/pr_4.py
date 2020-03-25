import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def result_of_operation(a, b, c):
    return (a and b) or (a and c)


def get_source_data():
    return np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])


def result_of_matrix():
    return np.array([result_of_operation(*i) for i in get_source_data()])


def tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for i in range(len(weights)):
        result = layers[i](np.dot(result, weights[i][0]) + weights[i][1])
    return result


def each_element_of_tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for _ in range(len(weights) - 1)]
    layers.append(sigmoid)
    for weight in range(len(weights)):
        len_current_weight = len(weights[weight][1])
        step_result = np.zeros((len(result), len_current_weight))
        for i in range(len(result)):
            for j in range(len_current_weight):
                sum = 0
                for k in range(len(result[i])):
                    sum += result[i][k] * weights[weight][0][k][j]
                step_result[i][j] = layers[weight](sum + weights[weight][1][j])
        result = step_result
    return result


def my_print(model, dataset):
    weights = [layer.get_weights() for layer in model.layers]
    print(weights)
    tensor_res = tensor_result(dataset, weights)
    each_el = each_element_of_tensor_result(dataset, weights)
    model_res = model.predict(dataset)
    print(tensor_res)
    print(model_res)
    print("Результат тензорного вычисления:")
    print(tensor_res)
    print("Результат вычисления каждого элемента")
    print(each_el)
    print("Результат прогона через обученную модель:")
    print(model_res)


def start():
    train_data = get_source_data()
    validation_data = result_of_matrix()
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(3,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    my_print(model, train_data)
    model.fit(train_data, validation_data, epochs=12, batch_size=1)
    my_print(model, train_data)


if __name__ == '__main__':
    start()
