import tensorflow as tf

filename = "data/train.tfrecord"
dataset = tf.data.TFRecordDataset(filename)
iter_data = dataset.make_one_shot_iterator()

features = {
    "index" : tf.FixedLenFeature(shape=(1, 57), dtype=tf.int64, default_value=None),
    "value" : tf.FixedLenFeature(shape=(1, 57), dtype=tf.float32),
    "target" : tf.FixedLenFeature(shape=(), dtype=tf.int64),
}
parsed_example = tf.parse_single_example(iter_data.get_next(), features)
with tf.Session() as sess:
    print(sess.run(parsed_example['target']))
