import tensorflow as tf

class DeepFM:
    def __init__(self, feature_size, field_size, model_conf, 
                 activation="relu", ckpt_path=None):
        self.feature_size = feature_size
        self.field_size = field_size
        self.config = model_conf
        self.activation = activation
        self.ckpt_path = ckpt_path
        self.init_weight()
        self.session = tf.Session()
    
    def __del__(self):
        self.session.close()
        print("session closed !")

    def activate(self, Z):
        if self.activation == "relu":
            A = tf.nn.relu(Z)
        elif self.activation == "tanh":
            A = tf.nn.tanh(Z)
        else:
            A = Z
        return A

    def init_weight(self):
        # init embedding matrix
        self.embedding = tf.get_variable("embedding",
                                         shape=(self.feature_size,
                                                self.config["embedding_size"]),
                                         dtype=tf.float32,
                                         initializer=tf.initializers.random_normal(seed=1))
        # init FM first order weight
        self.w_fm = tf.get_variable("weight_FM",
                                    shape=(self.field_size, 1),
                                    dtype=tf.float32,
                                    initializer=tf.initializers.random_normal(seed=1, stddev=0.01))

        # init first layer's param of MLP
        self.w_f_mlp = tf.get_variable("weight_first_MLP",
                                       shape=(self.config["embedding_size"] * self.field_size,
                                              self.config["layer_width"]),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.glorot_normal(seed=1))
        self.b_f_mlp = tf.get_variable("bias_first_MLP",
                                       shape=(1, self.config["layer_width"]),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.zeros())

        # init hidden layers' params of MLP
        self.w_h_mlp = tf.get_variable("weight_hidden_MLP",
                                       shape=(self.config["layer_depth"],
                                              self.config["layer_width"], self.config["layer_width"]),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.glorot_normal(seed=1))
        self.b_h_mlp = tf.get_variable("bias_hidden_MLP",
                                       shape=(self.config["layer_depth"],
                                              1, self.config["layer_width"]),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.zeros())

        # init last layer's params of DeepFM
        self.w_l_fm = tf.get_variable("weight_last_fm",
                                       shape=(self.config["embedding_size"] + self.field_size, 1),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.glorot_normal(seed=1))
        self.w_l_mlp = tf.get_variable("weight_last_mlp",
                                       shape=(self.config["layer_width"], 1),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.glorot_normal(seed=1))
        self.b_l = tf.get_variable("bias_last",
                                       shape=(1, 1),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.zeros())

    def _factorization_machine(self, x_index, x_value):
        # first order part
        # first_order = tf.squeeze(tf.matmul(x_value, self.w_fm), 1)
        first_order = tf.multiply(tf.transpose(self.w_fm, [1, 0]), x_value) # None * feild_size
        # print(first_order.shape)

        # second order part 
        vectors = tf.nn.embedding_lookup(self.embedding, x_index) # None * field_size * embedding_size
        x_value_ex = tf.expand_dims(x_value, 2) # None * field_size * 1
        v = tf.multiply(x_value_ex, vectors) # None * field_size * embedding_size

        # interaction_v = tf.matmul(v, tf.transpose(v, [0, 2, 1]))
        # interaction_self = tf.multiply(v, v)

        # second_order = 0.5 * \
        #     tf.reduce_sum(interaction_v, [1, 2]) - 0.5 * \
        #     tf.reduce_sum(interaction_self, [1, 2])
        
        # proof from "Factorization Machines" paper
        sum_square = tf.square(tf.reduce_sum(v, 1)) # None * K
        square_sum = tf.reduce_sum(tf.square(v), 1) # None * K
        second_order = 0.5 * tf.subtract(sum_square, square_sum) # None * K
        # print(second_order.shape)
        concat_f_s = tf.concat([first_order, second_order], axis=1)
        fm_out = tf.matmul(concat_f_s, self.w_l_fm)
        return fm_out

    def _mlp(self, x_index, x_value):
        dense_vectors = tf.nn.embedding_lookup(self.embedding, x_index)
        x_value = tf.expand_dims(x_value, 2)
        dense_vectors = tf.multiply(dense_vectors, x_value)
        input_vector = tf.reshape(dense_vectors, 
                                  shape=(-1, dense_vectors.shape[1] * dense_vectors.shape[2]))
        Z = tf.matmul(input_vector, self.w_f_mlp) + self.b_f_mlp
        A = self.activate(Z)
        for i in range(self.w_h_mlp.shape[0]):
            Z = tf.matmul(A, self.w_h_mlp[i]) + self.b_h_mlp[i]
            A = self.activate(Z)
        dnn_output = tf.matmul(A, self.w_l_mlp)
        return dnn_output

    def _forward_propagate(self, x_index, x_value):
        self.fm = self._factorization_machine(x_index, x_value)
        self.mlp = self._mlp(x_index, x_value)
        output = tf.squeeze(tf.nn.sigmoid(self.mlp + self.fm + self.b_l), 1)
        return output

    def loss(self, y_hat, y):
        binary_cross_entropy = - y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)) \
            - (1 - y) * tf.log(tf.clip_by_value(1 - y_hat, 1e-10, 1.0))
        loss = tf.reduce_mean(binary_cross_entropy)
        return loss

    def predict(self, x_index, x_value, threshold=0.5):
        x_index = tf.constant(x_index, dtype=tf.int32, shape=x_index.shape)
        x_value = tf.constant(x_value, dtype=tf.float32, shape=x_value.shape)
        threshold = tf.constant(threshold, dtype=tf.float32)
        output = self._forward_propagate(x_index, x_value)
        output = tf.cast(tf.greater(output, threshold), tf.float32)
        output_value = self.session.run(output)
        return output_value

    def evaluate(self, x_index, x_value, y):
        y_predict = self.predict(x_index, x_value)
        # print(y, y_predict, sep='\n')
        P = y[y_predict == 1.0]
        N = y[y_predict == 0.0]
        TP = P[P == 1.0].shape[0]
        FP = P.shape[0] - TP
        FN = N[N == 1.0].shape[0]
        # print("TP:", TP, "\tFP", FP, "\tFN", FN)
        F1 = 2*TP / (2*TP + FP + FN)
        return F1

    def train(self, dataset_train, dataset_val=None, epochs=120, 
              batch_size=128, learning_rate=0.001, 
              shuffle_size=1000, verbose=True, ckpt_path="./checkpoint/model"):
        dataset_val = dataset_val.batch(800)
        iter_val_data = dataset_val.make_one_shot_iterator()
        val = iter_val_data.get_next()
        x_index_val, x_value_val, y_val = self.session.run([val["index"], val["value"], val["target"]])
        x_index_val = tf.constant(x_index_val, dtype=tf.int32, shape=x_index_val.shape)
        x_value_val = tf.constant(x_value_val, dtype=tf.float32, shape=x_value_val.shape)
        y_val = tf.constant(y_val, dtype=tf.float32, shape=y_val.shape)
        loss_val = self.loss(self._forward_propagate(x_index_val, x_value_val), y_val)
        
        
        dataset_train = dataset_train.repeat(epochs).shuffle(shuffle_size).batch(batch_size)
        iter_train_data = dataset_train.make_one_shot_iterator()
        batch = iter_train_data.get_next()
        x_index, x_value, y = batch["index"], batch["value"], batch["target"]
        y_hat = self._forward_propagate(x_index, x_value)
        loss = self.loss(y_hat, y)

        opt = tf.train.AdamOptimizer().minimize(loss)

        init = tf.global_variables_initializer()
        self.session.run(init)
        
        i = 0
        while True:
            try:
                _, loss_value, value, index, y_value= \
                self.session.run([opt, loss, x_value, x_index, y])
                
                loss_value_val, value_val, index_val, y_value_val= \
                self.session.run([loss_val, x_value_val, x_index_val, y_val])
                if verbose and (i % 100 == 0):
                    print("step:", i)
                    print("train: -- loss:", loss_value, end="\t")
                    F1 = self.evaluate(index, value, y_value)
                    print("F1:%.4f" % F1)
                    print("val: -- loss:", loss_value_val, end="\t")
                    F1_val = self.evaluate(index_val, value_val, y_value_val)
                    print("F1:%.4f" % F1_val)
                i += 1
            except tf.errors.OutOfRangeError:
                break
        
        # save trained params
        saver = tf.train.Saver()
        saver.save(self.session, ckpt_path)
            
