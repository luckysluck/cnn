try:
    import urllib.request
except:
    raise ImportError('You need to update python')

import os
import pickle
import gzip
import numpy as np

url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir,  "mnist.pkl")
data_files = {
    "train_img" : "train-images-idx3-ubyte.gz",
    "train_labels" : "train-labels-idx1-ubyte.gz",
    "test_img" : "t10k-images-idx3-ubyte.gz",
    "test_labels" : "t10k-labels-idx1-ubyte.gz"
}

def _download(file):
    file_path = os.path.join(dataset_dir, file)
    if os.path.exists(file_path):
        return
    print("Downloading " + file)
    urllib.request.urlretrieve(url + file, file_path)
    print("Done")

def download_mnist():
    for item in data_files.values():
        _download(item)

def _parse_label(file):
    file_path = os.path.join(dataset_dir,  file)
    print("Parsing " + file)
    with gzip.open(file_path, 'rb') as f:
        f_content = f.read()
        labels = np.frombuffer(f_content, np.uint8, offset = 8)
    return labels

def _parse_img(file):
    file_path = os.path.join(dataset_dir, file)
    print("Parsing " + file)
    with gzip.open(file_path, 'rb') as f:
        f_content = f.read()
        img = np.frombuffer(f_content, np.uint8, offset = 16)
    return img.reshape((-1, 784))

def parse_mnist():
    dataset = dict()
    dataset["train_img"] = _parse_img(data_files["train_img"])
    dataset["train_labels"] = _parse_label(data_files["train_labels"])
    dataset["test_img"] = _parse_img(data_files["test_img"])
    dataset["test_labels"] = _parse_label(data_files["test_labels"])
    return dataset

def _one_hot_change(labels):
    t = np.zeros((labels.size, 10))
    for idx, label in enumerate(labels):
        t[idx, label] = 1
    return t

def init_mnist():
    download_mnist()
    data = parse_mnist()
    with open(save_file, 'wb') as f:
        pickle.dump(data, f, -1)

def load_mnist(normalize = True, flatten = False, one_hot = False):
    with open(save_file, 'rb') as f:
        data = pickle.load(f)
    if normalize:
        for key in ("train_img", "test_img"):
            data[key] = data[key].astype(np.float32) / 255.0
    if one_hot:
        data['train_labels'] = _one_hot_change(data['train_labels'])
        data['test_labels'] = _one_hot_change(data['test_labels'])
    if not flatten:
        for key in ("train_img", "test_img"):
            data[key] = data[key].reshape((-1, 1, 28, 28))
    return (data['train_img'], data['train_labels']), (data['test_img'], data['test_labels'])


if __name__ == '__main__':
    init_mnist()