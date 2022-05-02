import itertools
import numpy as np
from typing import Any

filename = 'iris.data.txt'


def ext_check(__, _) -> None:
    if __.endswith('.csv'):
        _.readline()


def test_case(training_data, training_label, testing_data, testing_label) -> None:
    result_set = []
    correct_rate = 0.0
    error_rate = 0.0
    ratio = len(testing_data)

    for testing_set in testing_data:
        result = classify(testing_set, training_data, training_label, 3)
        result_set.append(result)

    for i in range(30):
        print(f'Prediction: {result_set[i]}  Real value: {testing_label[i]}')

        if result_set[i] != testing_label[i]:
            error_rate += 1.0
        else:
            correct_rate += 1.0

    error_rate = error_rate / float(ratio)
    correct_rate = correct_rate / float(ratio)

    print(f'Correct count: {correct_rate.__format__(".2f")} \tError count: {error_rate.__format__(".2f")}')


def classify(inX, dataset, labels, k) -> list:
    size = dataset.shape[0]
    matrix = (np.tile(inX, (size, 1)) - dataset) ** 2
    dist = matrix.sum(axis=1) ** 0.5
    sorted_dist = dist.argsort()
    count = {}

    for i in range(k):
        label = labels[sorted_dist[i]]
        count[label] = count.get(label, 0) + 1
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)

    return sorted_count[0][0]


def line_size() -> int:
    with open(filename) as f:
        line = f.readline().split(',')
        return len(line)


def normalize_data(dataset) -> np.ndarray:
    minimum = dataset.min(axis=0)
    maximum = dataset.max(axis=0)
    ranges = maximum - minimum
    data_size = dataset.shape[0]
    normalized = dataset - np.tile(minimum, (data_size, 1))
    normalized = normalized / np.tile(ranges, (data_size, 1))
    return normalized


def prepare_norm(file_size, cols):
    data = np.zeros((file_size, cols))
    labels = []

    with open(filename) as f:
        idx = 0
        ext_check(filename, f)

        for line in f:
            file_list = line.strip().split(',')
            data[idx] = [val for val in file_list[:cols]]
            labels.append(file_list[-1])
            idx += 1

        data = normalize_data(data)
        labels = np.reshape(labels, (file_size, 1))
        norm_data = np.concatenate((data, labels), axis=1)

    return norm_data


def data_split(file, ratio, rand=False) -> [np.ndarray, [Any], np.ndarray, [Any]]:
    file_size = len(file.readlines())
    elem = line_size()
    cols = elem - 1
    testing = int(file_size * ratio)
    training = file_size - testing
    training_data = np.zeros((training, cols))
    training_label = []
    testing_data = np.zeros((testing, cols))
    testing_label = []

    normalized = prepare_norm(file_size, cols)

    if rand:
        np.random.shuffle(normalized)

    train_idx = 0
    for line in itertools.islice(normalized, training):
        file_list = line
        training_data[train_idx] = [float(val) for val in file_list[:cols]]
        training_label.append(file_list[-1])
        train_idx += 1

    test_idx = 0
    for line in itertools.islice(normalized, training, file_size):
        file_list = line
        testing_data[test_idx] = [float(val) for val in file_list[:cols]]
        testing_label.append(file_list[-1])
        test_idx += 1

    return training_data, training_label, testing_data, testing_label


def main() -> None:
    with open(filename) as f:
        ext_check(filename, f)
        training_data, training_label, testing_data, testing_label = data_split(file=f, ratio=0.2)
    test_case(training_data, training_label, testing_data, testing_label)


if __name__ == '__main__':
    main()
