from cProfile import label
import pandas as pd
import numpy as np

def load_bank(path):
    n_class = 20
    df = pd.read_csv(path, sep=';')
    cat_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]
    for i in cat_index:
        df.iloc[:, i] = df.iloc[:, i].astype('category').cat.codes
    data = df.iloc[:, :n_class].values[1:].astype(np.float32)
    labels = df.iloc[:, n_class].astype('category').cat.codes.values[1:].astype(np.int32)
    return data, labels

if __name__ == '__main__':
    data, labels = load_bank('./dataset/Bank/bank-additional-full.csv')
    data = np.array(data)
    labels = np.array(labels)
    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    random_idx = np.random.choice(np.arange(data.shape[0]), 3000, replace=False)
    data = data[random_idx]
    labels = labels[random_idx]
    np.save('./dataset/Bank/bank_data.npy', data)
    np.save('./dataset/Bank/bank_label.npy', labels)
    print(data.shape, data.max(), data.min())
    print(labels.shape)