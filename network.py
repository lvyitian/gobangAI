import tensorflow as tf
import numpy as np
import random

from data_structure import *
from game_tree import *


def add_layer(input, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random.normal([in_size, out_size]))
    # baises = tf.Variable(tf.zeros([1, out_size]) + 1e-9)
    calc = tf.matmul(input, weight)
    if activation_function is None:
        return calc
    else:
        return activation_function(calc)


def flat_cbmap(cbmap):
    flatten = []
    for i in cbmap:
        flatten.extend(i)
    return flatten


def train(datanum=2000, iternum=2000, old_net=True):
    sess = tf.compat.v1.Session()

    learning_rate = 1e-6
    hidden_notes1 = 10

    tf.compat.v1.disable_eager_execution()
    xs = tf.compat.v1.placeholder(tf.float32, [None, 225])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # hidden_layer1 = add_layer(xs, 225, hidden_notes1)
    # hidden_layer2 = add_layer(hidden_layer1, 100, 20, tf.nn.relu)
    # prediction = add_layer(hidden_layer1, hidden_notes1, 1)

    weight1 = tf.Variable(tf.random.normal([225, hidden_notes1]))
    weight2 = tf.Variable(tf.random.normal([hidden_notes1, 1]))

    hidden_layer1 = tf.matmul(xs, weight1)
    prediction = tf.matmul(hidden_layer1, weight2)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), [0]))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    if old_net is True:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "net\\network.ckpt")

    x_data = []
    y_data = []

    for i in range(datanum):
        coords = data_sl.load("test\\coords\\coord" + str(i) + ".dat")
        cbmap = create_cbmap(coords)
        x_data.append(flat_cbmap(cbmap))
        y_data.append([evaluate(cbmap) / 1e7])

    for i in range(iternum):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % (iternum / 100) == 0:
            print("step" + str(i) + ": " + str(sess.run(loss, feed_dict={xs: x_data, ys: y_data})))
            # print(sess.run(weight))
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "net\\network.ckpt")
    sess.close()


def genetic_train():
    hidden_notes1 = 5

    data_num = 200
    pop_num = 10
    iter_num = 100
    gen_num = 200

    select_rate = 0.5
    recombine_rate = 1.0
    mutate_rate = 0.1


    x_data = []
    y_data = []

    for i in range(data_num):
        coords = data_sl.load("test\\coords\\coord" + str(i) + ".dat")
        cbmap = create_cbmap(coords)
        x_data.append(flat_cbmap(cbmap))
        y_data.append([evaluate(cbmap) / 1e7])

    tf.compat.v1.disable_eager_execution()
    xs = tf.compat.v1.placeholder(tf.float32, [None, 225])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])

    weight1 = tf.Variable(tf.random.normal([225, hidden_notes1]))
    weight2 = tf.Variable(tf.random.normal([hidden_notes1, 1]))

    hidden_layer1 = tf.matmul(xs, weight1)
    prediction = tf.matmul(hidden_layer1, weight2)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), [0]))
    LEARNING_RATE_BASE = 1e-3  # 最初学习率
    LEARNING_RATE_DECAY = 0.985  # 学习率的衰减率
    LEARNING_RATE_STEP = iter_num * pop_num  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
    gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
    learning_rate = tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE
                                               , gloabl_steps,
                                               LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, gloabl_steps)

    init = tf.compat.v1.global_variables_initializer()

    weights = []

    sess = tf.compat.v1.Session()

    for pop in range(pop_num):
        sess.run(init)

        saver = tf.compat.v1.train.Saver()
        # saver.restore(sess, "net\\network" + str(pop) + "\\net.ckpt")

        for iter in range(iter_num):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        lost = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        # print("第" + str(pop) + "个体: " + str(lost))
        weights.append([sess.run(weight1), sess.run(weight2), lost])

        saver.save(sess, "net\\network" + str(pop) + "\\net.ckpt")

    for gen in range(1, gen_num):
        remove_num = int((1-select_rate) * pop_num)
        weights = sorted(weights, key=lambda x: x[2], reverse=True)
        sum = 0.0
        for i in weights:
            sum += i[2]
        avg = sum / pop_num
        print("第" + str(gen) + "代: 平均误差 " + str(avg) + " 最小误差 " + str(weights[-1][2]) + " 学习率 " + str(sess.run(learning_rate)))
        for i in range(remove_num):
            parent1, parent2 = random.sample(range(remove_num, pop_num), 2)
            weights[i] = weights[parent1]
            if random.uniform(0, 1) < recombine_rate:
                chosen1 = random.sample(range(0, 224), 112)
                chosen2 = random.sample(range(0, hidden_notes1-1), hidden_notes1//2)
                for k in chosen1:
                    weights[i][0][k] = weights[parent2][0][k]
                for k in chosen2:
                    weights[i][1][k] = weights[parent2][1][k]
            if random.uniform(0, 1) < mutate_rate:
                chosen1 = random.sample(range(0, 224), 112)
                chosen2 = random.sample(range(0, hidden_notes1-1), hidden_notes1//2)
                tmp = weights[i][0][chosen1[0]]
                weights[i][0][chosen1[0]] = weights[i][0][chosen1[1]]
                weights[i][0][chosen1[1]] = tmp
                tmp = weights[i][1][chosen2[0]]
                weights[i][1][chosen2[0]] = weights[i][1][chosen2[1]]
                weights[i][1][chosen2[1]] = tmp

        for i in range(len(weights)):

            saver = tf.compat.v1.train.Saver()

            weight1.load(weights[i][0], sess)
            weight2.load(weights[i][1], sess)

            for iter in range(iter_num):
                sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

            lost = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            # print("第" + str(i) + "个体: " + str(lost))
            weights[i] = [sess.run(weight1), sess.run(weight2), lost]

            saver.save(sess, "net\\network" + str(i) + "\\net.ckpt")


def test(coord_num):
    learning_rate = 1e-3
    hidden_notes1 = 5
    tf.compat.v1.disable_eager_execution()
    xs = tf.compat.v1.placeholder(tf.float32, [None, 225])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])

    weight1 = tf.Variable(tf.random.normal([225, hidden_notes1]))
    weight2 = tf.Variable(tf.random.normal([hidden_notes1, 1]))

    hidden_layer1 = tf.matmul(xs, weight1)
    prediction = tf.matmul(hidden_layer1, weight2)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), [0]))
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()

    sess = tf.compat.v1.Session()
    sess.run(init)

    x_data = []
    coords = data_sl.load("test\\coords\\coord" + str(coord_num) + ".dat")
    cbmap = create_cbmap(coords)
    x_data.append(flat_cbmap(cbmap))

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "net\\network9.ckpt")

    print(1e7 * (sess.run(prediction, feed_dict={xs: x_data}))[0][0])
    print(evaluate(cbmap))


def main():
    # genetic_train()
    # test(5)
    pass


main()
