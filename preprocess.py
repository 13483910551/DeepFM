import tensorflow as tf
import pandas as pd

path_train = "data/train.csv"
path_test = "data/test.csv"

def _preprocess(path_train, path_test):
    # TODO
    return None


def load_data():
    data_train = Data.from_csv(path_train)
    data_test = Data.from_csv(path_test)
    dataset_tuple = (data_train.get_dataset(), data_test.get_dataset())
    return dataset_tuple

class Data:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    @staticmethod
    def from_csv(path):
        dataset = tf.data.TextLineDataset(path)
        dataset = dataset.skip(1)
        return Data(dataset)

if __name__ == "__main__":
    # dataset = load_data()
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    # with tf.Session() as sess:
    #     for i in range(10):
    #         print(sess.run(next_element))
    pass