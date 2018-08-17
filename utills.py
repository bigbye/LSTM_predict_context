import collections
import os
import sys
import argparse  # 参数解析
import tensorflow as tf
import datetime
import numpy as np

Py3 = sys.version_info[0] == 3

# 数据集的目录
data_path = "data"

# 保存训练所得的模型参数文件的目录
save_path = os.getcwd() + os.sep + "save"

# 测试时读取模型参数文件的名称
load_file = "train-checkpoint-69"  # 倒数第二个模型


def generate_batches(data, batch_size, n_steps):
    raw_data = tf.convert_to_tensor(data, tf.int32)
    n_data = tf.size(raw_data)
    n_batches = n_data // batch_size

    b_data = tf.reshape(raw_data[0:batch_size * n_batches], [batch_size, n_batches])
    step_size = (n_batches - 1) // n_steps  # step_size的计数单位是batch
    # range_input_producer 可以用多线程异步的方式从数据集里提取数据
    # 用多线程可以加快训练，因为feed_dict的赋值方式效率不高
    # shuffle为False表示不打乱数据而按照队列先进先出的方式提取数据
    i = tf.train.range_input_producer(step_size, shuffle=False).dequeue()
    x = b_data[:, i * n_steps:(i + 1) * n_steps]
    x.set_shape([batch_size, n_steps])

    y = b_data[:, i * n_steps + 1:(i + 1) * n_steps + 1]
    y.set_shape([batch_size, n_steps])

    return x, y


class Input(object):
    # 数据按step_size分为多个step，每个step又有多个batch。
    def __init__(self, batch_size, n_steps, data):
        # tf_data = tf.convert_to_tensor(data, tf.int32)
        # n_data = len(data)
        # n_batches = n_data // batch_size  # //表示整除
        # self.data = tf.reshape(tf_data[0:batch_size * n_batches], [batch_size, n_batches])
        # step_size = n_batches // n_steps
        # # 多线程异步的方式从数据集提取数据，提高效率
        # i = tf.train.range_input_producer(step_size, shuffle=False).dequeue()
        # x = self.data[:, i * n_steps:(i + 1) * n_steps]
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.step_size = (len(data) // batch_size - 1) // n_steps  # 一步包含多个batch
        self.input_data, self.targets = generate_batches(data, batch_size, n_steps)


def read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        if Py3:
            return f.read().replace('\n', '<eos>').split()
        else:
            return f.read().decode('utf-8').replace('\n', '<eos>').split()


# 构造从单词到唯一整数值的映射
def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    counter_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*counter_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data(data_path):
    if not os.path.exists(data_path):
        raise Exception("包含所有数据集文件的 {} 文件夹 不在此目录下，请添加".format(data_path))

    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = build_vocab(train_path)  # train path 包含了所有单词，valid_data,test_data是train_data的子集

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    vocab_size = len(word_to_id)

    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    # print(word_to_id)
    # print("===============")
    # print(vocab_size)
    # print("===============")
    # print(train_data[:10])
    # print("===============")
    # print(" ".join([id_to_word[x] for x in train_data[:10]]))
    # print("===============")
    # print(id_to_word)
    return train_data, valid_data, test_data, vocab_size, id_to_word


if __name__ == '__main__':
    load_data(data_path)
