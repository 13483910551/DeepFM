import tensorflow as tf
import config
from model import DeepFM

train_file = "data/train.tfrecord"
val_file = "data/val.tfrecord"

def parsing_record(record):
    features = {
    "index" : tf.FixedLenFeature(shape=(57,), dtype=tf.int64, default_value=None),
    "value" : tf.FixedLenFeature(shape=(57,), dtype=tf.float32),
    "target" : tf.FixedLenFeature(shape=(), dtype=tf.float32),
    }
    parsed_example = tf.parse_single_example(record, features)
    return parsed_example
def main():
    dataset_train = tf.data.TFRecordDataset(train_file)
    dataset_val = tf.data.TFRecordDataset(val_file)

    dataset_train = dataset_train.map(parsing_record)
    dataset_val = dataset_val.map(parsing_record)
    
    deepFM = DeepFM(config.data_size["feature_size"], 
                    config.data_size["field_size"], config.model_conf)
    deepFM.train(dataset_train, dataset_val=dataset_val, epochs=500, batch_size=128)

if __name__ == "__main__":
    main()


