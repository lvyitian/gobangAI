import sys
from collections import deque
import random

import pygame

from data_structure import data_sl
from game_tree import max_search
from gobang_system import Gobang
from network import flat_cbmap
import numpy as np

import tensorflow as tf


def flat_cbmap_reverse(cbmap):
    cbmap = cbmap[:]
    for i in range(len(cbmap)):
        cbmap[i] = -cbmap[i]
    return cbmap


def cbmap_reverse(cbmap):
    cbmap = [row[:] for row in cbmap]
    for i in range(15):
        for j in range(15):
            cbmap[i][j] = -cbmap[i][j]
    return cbmap

tf.compat.v1.disable_eager_execution()
xs = tf.compat.v1.placeholder(tf.float32, [None, 15, 15])
ys = tf.compat.v1.placeholder(tf.float32, [None, 1])
Q_value = 0
Q_weights = 0


def create_Q_network():
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

    global Q_value
    global Q_weights
    global xs
    Q_value = conv_basic(xs, weights, biases)
    Q_Weights = [weights, biases]


def conv_basic(_input, _w, _b):
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


def flip(cbmap, type):
    cbmap = [row[:] for row in cbmap]
    if type == 1:
        for i in range(7):
            for j in range(15):
                tmp = cbmap[i][j]
                cbmap[i][j] = cbmap[14-i][j]
                cbmap[14 - i][j] = tmp
    if type == 2:
        for i in range(15):
            for j in range(7):
                tmp = cbmap[i][j]
                cbmap[i][j] = cbmap[i][14 - j]
                cbmap[i][14- j] = tmp
    if type == 3:
        for i in range(14):
            for j in range(14 - i):
                tmp = cbmap[i][j]
                cbmap[i][j] = cbmap[14 - i][14 - j]
                cbmap[14 - i][14 - j] = tmp
    if type >= 4:
        for i in range(14):
            for j in range(14-i):
                tmp = cbmap[i][j]
                cbmap[i][j] = cbmap[14-j][14-i]
                cbmap[14 - j][14 - i] = tmp
        if 8 > type > 4:
            cbmap = flip(cbmap, type-4)
    return cbmap

def reinforce_train():
    global Q_value
    global Q_weights
    global xs
    global ys
    # 设置
    hidden_notes1 = 1000  # 隐藏层结点数
    queue_max = 50000  # 最多存放的经验回放数
    greedy_rate = 0.95  # 选择最优Q值的概率
    learning_rate = 0.9

    batch_num = 1000  # 每次训练网络使用的数据数
    iter_num = 1  # 每次数据迭代次数

    # 载入数据
    # dataset = deque(maxlen=queue_max)
    dataset = data_sl.load("net\\reinforce\\dataset.dat")

    # 初始化游戏界面
    gobang = Gobang()
    pygame.init()
    screen = pygame.display.set_mode((535, 535), 0, 32)
    clock = pygame.time.Clock()

    # 初始化神经网络


    # weight1 = tf.Variable(tf.random.normal([225, hidden_notes1]))
    # weight2 = tf.Variable(tf.random.normal([hidden_notes1, 1]))
    #
    # bias1 = tf.Variable(tf.ones([1, hidden_notes1] ))
    # bias2 = tf.Variable(tf.ones([1, 1]))
    #
    # hidden_layer1 = tf.matmul(xs, weight1) + bias1
    # prediction = tf.matmul(hidden_layer1, weight2) + bias2
    create_Q_network()

    loss = tf.reduce_mean(tf.compat.v1.squared_difference(Q_value,ys))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()

    sess = tf.compat.v1.Session()

    sess.run(init)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "net\\reinforce\\reinforce.ckpt")

    count = 0
    recent_loss = deque(maxlen=20)
    while True:
        for event in pygame.event.get():
            clock.tick(20)
            if event.type == pygame.QUIT:
                sys.exit()  # 无限循环更新

        # 模拟对弈
        gobang.init()
        temp_data = {"map": [], "score": []}
        gobang.drop_at(7, 7)
        temp_data["map"].append(cbmap_reverse(gobang.cbmap))
        temp_data["score"].append(1.0)
        while True:
            # if gobang.black_turn:
            #     for event in pygame.event.get():
            #         clock.tick(20)
            #         if event.type == pygame.QUIT:
            #             sys.exit()  # 无限循环更新
            #         elif event.type == pygame.MOUSEBUTTONDOWN:
            #             if event.button == 1:  # 按下的是鼠标左键
            #                 i, j = gobang.get_coord(event.pos)  # 将物理坐标转换成矩阵的逻辑坐标
            #                 if gobang.check_at(i, j):  # 检查(i,j)位置能否被占用，如未被占用返回True
            #                     cbmap = [row[:] for row in gobang.cbmap]
            #                     cbmap = cbmap_reverse(cbmap)
            #                     gobang.drop_at(i, j)
            #
            #                     selectable = []
            #                     positions = []
            #                     for i in range(15):
            #                         for j in range(15):
            #                             if cbmap[i][j] == 0:
            #                                 pred = np.copy(cbmap)
            #                                 pred[i][j] = 1
            #                                 selectable.append(pred)
            #                                 positions.append([i, j])
            #
            #                     pred_q = sess.run(Q_value, feed_dict={xs: selectable})
            #                     max_value = -1e9
            #                     best_pos = [0, 0]
            #                     for i in range(len(positions)):
            #                         value = pred_q[i] + random.uniform(0, 0.01)  # 如果没有最优步子 则随机选择一步
            #                         if value > max_value:
            #                             max_value = value
            #                             best_pos = positions[i]
            #                     cbmap = [row[:] for row in gobang.cbmap]
            #                     cbmap = cbmap_reverse(cbmap)
            #                     temp_data["map"].append(cbmap)
            #                     temp_data["score"].append(max_value)
            # # 以一定概率选择已知最优解下

            # if gobang.black_turn:
            #     cbmap = cbmap_reverse(cbmap)
            cbmap = [row[:] for row in gobang.cbmap]
            if gobang.black_turn:
                cbmap = cbmap_reverse(cbmap)
            selectable = []
            positions = []
            for i in range(15):
                for j in range(15):
                    if cbmap[i][j] == 0:
                        pred = np.copy(cbmap)
                        pred[i][j] = 1
                        selectable.append(pred)
                        positions.append([i, j])

            pred_q = sess.run(Q_value, feed_dict={xs: selectable})
            max_value = -1e9
            best_pos = [0, 0]
            for i in range(len(positions)):
                value = pred_q[i] * random.uniform(0, 0.01)  # 如果没有最优步子 则随机选择一步
                if value > max_value:
                    max_value = value
                    best_pos = positions[i]
            # 一定概率随机落子
            if random.uniform(0, 1) > greedy_rate:
                step = random.randint(0, len(positions)-1)
                best_pos = positions[step]
            if random.uniform(0, 1) < 0.1:
                cbmap[best_pos[0]][best_pos[1]] = 1
                gobang.drop_at(best_pos[0], best_pos[1])
            else:
                value, x, y = max_search(gobang.cbmap, 1, 1e10, -1e10)
                cbmap[x][y] = 1
                gobang.drop_at(x, y)
            temp_data["map"].append(cbmap)
            temp_data["score"].append(max_value)
            # else:
            #     cbmap = flat_cbmap_reverse(cbmap)
            #     if random.uniform(0, 1) < 0.0:
            #         value, x, y = max_search(gobang.cbmap, 1, 1e10, -1e10)
            #         gobang.drop_at(x, y)
            #         cbmap[15 * x + y] = 1
            #     else:
            #         selectable = []
            #         for i in range(len(cbmap)):
            #             if cbmap[i] == 0:
            #                 selectable.append(i)
            #         si = random.randint(0, len(selectable) - 1)
            #         besti = selectable[si]
            #         cbmap[besti] = 1
            #         gobang.drop_at(besti // 15, besti % 15)
            # for event in pygame.event.get():
            #     clock.tick(20)
            #     if event.type == pygame.QUIT:
            #         sys.exit()  # 无限循环更新
            #     elif event.type == pygame.MOUSEBUTTONDOWN:
            #         if event.button == 1:  # 按下的是鼠标左键
            #             i, j = gobang.get_coord(event.pos)  # 将物理坐标转换成矩阵的逻辑坐标
            #             if gobang.check_at(i, j):  # 检查(i,j)位置能否被占用，如未被占用返回True
            #                 gobang.drop_at(i, j)
            #                 cbmap[15 * i + j] = 1

            screen.blit(gobang.chessboard(), (0, 0))
            pygame.display.update()
            if gobang.check_over():
                temp_data["score"][-1] = 1.0
                break
            elif gobang.check_full():
                temp_data["score"][-1] = 0.0
                break
        count += 1
        # 回顾棋局
        for i in range(1, len(temp_data["score"]) - 2):
            temp_data["score"][i] = - temp_data["score"][i + 1] * learning_rate
        for i in range(len(temp_data)):
            for k in range(1, 8):
                cb = flip(temp_data["map"][i], k)
                temp_data["map"].append(cb)
                temp_data["score"].append(temp_data["score"][i])
        # print(temp_data["score"])

        # 合并入数据集
        for i in range(len(temp_data["score"])):
            dataset.append([temp_data["map"][i], temp_data["score"][i]])

        # 保存数据库
        data_sl.save(dataset, "net\\reinforce\\dataset.dat")
        # 抽取数据训练网络
        train_data = []
        if batch_num < len(dataset):
            train_data = random.sample(dataset, k=batch_num)
        else:
            train_data = dataset
        for i in range(len(temp_data["score"])):
            train_data.append([temp_data["map"][i], temp_data["score"][i]])
        x_data = []
        y_data = []
        for cbmap, score in train_data:
            x_data.append(cbmap)
            y_data.append([score])

        for iter in range(iter_num):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

        last_loss = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        recent_loss.append(last_loss)
        avg_loss = sum(recent_loss) / len(recent_loss)
        print("第" + str(count) + "局: 误差 " + str(last_loss) + " 最近" + str(len(recent_loss)) + "次平均误差 " + str(avg_loss))

        sum(recent_loss)

        # print(sess.run(weight2))
        # print(sess.run(bias1))
        saver.save(sess, "net\\reinforce\\reinforce.ckpt")


def main():
    reinforce_train()


main()
