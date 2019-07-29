import tensorflow as tf
import pandas as pd
from itertools import chain
from config import data_descr


class Data:
    def __init__(self, config):
        self.config = config
        self._load_data()

    def _load_data(self):
        self.df_train = pd.read_csv(self.config["path_train"])
        self.df_test = pd.read_csv(self.config["path_test"])

    def _transform(self, df):
        cat_feature_size = {col: df[col].unique().shape[0]
                            for col in self.config["cat_cols"]}
        feature_start = {}
        feature_total_size = 0
        feature_total_cols = df.columns[2:]

        for col in feature_total_cols:
            feature_start[col] = feature_total_size
            if col in self.config["cat_cols"]:
                feature_total_size += cat_feature_size[col]
            else:
                feature_total_size += 1

        self.feature_size = feature_total_size
        self.filed_size = len(feature_total_cols)
        x_index = df[feature_total_cols].copy()

        for col in feature_total_cols:
            if col in self.config["cat_cols"]:
                x_index[col] += feature_start[col]
            else:
                x_index[col] = feature_start[col]

        means = df[self.config["num_cols"]].mean()
        stds = df[self.config["num_cols"]].std()
        norm = (df[self.config["num_cols"]] - means) / stds

        df.loc[:, self.config["cat_cols"]] = 1
        df.loc[:, self.config["num_cols"]] = norm

        x_value = df[feature_total_cols]
        y = df['target']
        return x_index, x_value, y

    def to_TFRecord(self, filename):
        x_index, x_value, y = self._transform(self.df_train)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index, value, label in zip(x_index.values, x_value.values, y.values):
                features = {}
                features["index"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=index))
                features["value"] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=value))
                features["target"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))
                tf_features = tf.train.Features(feature=features)
                tf_example = tf.train.Example(features=tf_features)
                tf_serialized = tf_example.SerializeToString()
                writer.write(tf_serialized)

if __name__ == "__main__":
    data = Data(data_descr)
    data.to_TFRecord(data_descr["data_path"]+"/train.tfrecord")
