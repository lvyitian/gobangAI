import tensorflow as tf




class evaluator(object):
    def create_Q_network(self):
        wc1 = tf.Variable(tf.random.normal([3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='wc1')
        wc2 = tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1), dtype=tf.float32, name='wc2')
        wc3 = tf.Variable(tf.random.normal([3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='wc3')
        wd1 = tf.Variable(tf.random.normal([256, 128], stddev=0.1), dtype=tf.float32, name='wd1')
        wd2 = tf.Variable(tf.random.normal([128, 1], stddev=0.1), dtype=tf.float32, name='wd2')

        bc1 = tf.Variable(tf.random.normal([64], stddev=0.1), dtype=tf.float32, name='bc1')
        bc2 = tf.Variable(tf.random.normal([128], stddev=0.1), dtype=tf.float32, name='bc2')
        bc3 = tf.Variable(tf.random.normal([256], stddev=0.1), dtype=tf.float32, name='bc3')
        bd1 = tf.Variable(tf.random.normal([128], stddev=0.1), dtype=tf.float32, name='bd1')
        bd2 = tf.Variable(tf.random.normal([1], stddev=0.1), dtype=tf.float32, name='bd2')

        weights = {
            'wc1': wc1,
            'wc2': wc2,
            'wc3': wc3,
            'wd1': wd1,
            'wd2': wd2
        }

        biases = {
            'bc1': bc1,
            'bc2': bc2,
            'bc3': bc3,
            'bd1': bd1,
            'bd2': bd2
        }

        self.Q_value = self.conv_basic(self.xs, weights, biases)
        Q_Weights = [weights, biases]

    def conv_basic(self, _input, _w, _b):
        # input
        _out = tf.reshape(_input, shape=[-1, 15, 15, 1])
        # conv layer 1
        _out = tf.nn.conv2d(_out, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc1']))
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv layer2
        _out = tf.nn.conv2d(_out, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc2']))
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv layer3
        _out = tf.nn.conv2d(_out, _w['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc3']))
        _out = tf.reduce_mean(_out, [1, 2])
        # fully connected layer1
        _out = tf.nn.relu(tf.add(tf.matmul(_out, _w['wd1']), _b['bd1']))
        # fully connected layer2
        _out = tf.add(tf.matmul(_out, _w['wd2']), _b['bd2'])
        return _out

    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.xs = tf.compat.v1.placeholder(tf.float32, [None, 15, 15])
        ys = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.create_Q_network()

        loss = tf.reduce_mean(tf.compat.v1.squared_difference(self.Q_value, ys))
        train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        self.sess = tf.compat.v1.Session()

        self.sess.run(init)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, "net\\reinforce\\reinforce.ckpt")

    def evaluate(self, cbmap):
        return self.sess.run(self.Q_value, feed_dict={self.xs: [cbmap]})[0][0]
