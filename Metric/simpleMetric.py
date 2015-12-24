from numpy as np
from common import load_mnist
from knn import KNN_predict

if __name__ == '__main__':
    train_data, test_data = load_mnist(percentage=0.01, skip_valid=True)
    train_x, train_y = train_data
    test_x, test_y = test_data


