import tensorflow as tf

class Model:
    def __init__(self, feature_index_dict, model_conf):
        self.feature_index_dict = feature_index_dict
        self.config = model_conf
        self.init_meta_info()
        #self.init_weight()

    def init_meta_info(self):
        self.field_size = len(self.feature_index_dict)
        self.feature_size = sum(map(lambda value: len(value), self.feature_index_dict.values()))
        feature_sorted = enumerate(sorted(self.feature_index_dict.items(), key=lambda x: x[1][0]))
        self.feature_seq_dict = {key: value[0] for key, value in feature_sorted}
    
    def init_weight(self):
        # init embedding matrix
        self.embedding = tf.get_variable("embedding",
                                         shape=(self.feature_size, self.config["embedding_size"]),
                                         dtype=tf.float32,
                                         initializer=tf.initializers.random_normal(seed=1))
        # init FM first order weight
        self.w_fm = tf.get_variable("weight_FM",
                                    shape=(self.feature_size, 1),
                                    dtype=tf.float32,
                                    initializer=tf.initializers.random_normal(seed=1))

    def factorization_machine(self, x_index, x_value):
        pass
        

    def forward_propagate(self, x_index, x_value):
        pass

    def train_batch(x_index, x_value, y):
        pass


if __name__ == "__main__":
    dict = {
        "test1" : [0, 1, 2],
        "test3" : [4],
        "test2" : [3]   
    }
    model = Model(dict, None)
    X = tf.constant([[0, 1.5, 2.5], [2, 1.1, 1.7], [1, 0.8, 0.7]], dtype=tf.float32, shape=(3, 3))
    model.preprocess_X(X)
    print(model.X_index)
    print(model.X_value)

