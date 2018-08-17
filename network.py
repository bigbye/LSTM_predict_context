import tensorflow as tf

'''
input_obj:输入参数
is_training:是否在训练\测试
hidden_size:LSTM层的神经元数目650
vocab_size:单词数目10000
num_layer:LSTM层数2
dropout:保留率0.5
init_scale:权重参数的上下限

'''


class Model(object):
    def __init__(self, input_obj, is_training, hidden_size, vocab_size, n_layers, dropout=0.5, init_scale=0.5):
        self.is_training = is_training
        self.input_obj = input_obj
        self.hidden_size = hidden_size
        self.batch_size = input_obj.batch_size
        self.n_steps = input_obj.n_steps  # 输入单词的个数，即LSTM展开的序列数

        # 让这里的操作和变量用CPU来计算，因为暂时还没有GPU的实现
        with tf.device("/cpu:0"):

            # 创建 词向量（word embedding），embedding表示dense vector（密集向量）
            # 词向量本质上是一种单词聚类（clustering）的方法 从10000个one-hot编码映射到词向量
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            # embedding_lookup 返回词向量 inputs
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        # 如果是训练时并且存在dropout 就进行dropout
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # 状态的存储和提取 4维
        # 第二维是2是因为对每一个LSTM单元有两个来自上一单元的输入
        # 一个是前一时刻LSTM的输出h(t-1)
        # 一个是前一时刻的单元状态C(t-1)
        # 这个c和h适用于构建之后的tf.contrib.rnn.LSTMStateTuple
        self.init_state = tf.placeholder(tf.float32, [n_layers, 2, self.batch_size, self.hidden_size])

        # 每一层的状态，在第0维展开
        state_per_layer_list = tf.unstack(self.init_state, axis=0)

        # 初始的状态（包含 前一时刻的LSTM输出和前一时刻的单元状态，用于之后的dynamic_rnn) 分两层分别构建
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in
             range(n_layers)])

        # 创建一个LSTM层，其中的神经元数目是hidden_size个（默认650）
        cell = tf.contrib.rnn.LSTMCell(hidden_size)

        # 如果是训练时 并且 Dropout 率小于 1，给 LSTM 层加上 Dropout 操作
        # 这里只给 输出 加了 Dropout 操作，留存率(output_keep_prob)是 0.5
        # 输入则是默认的 1，所以相当于输入没有做 Dropout 操作
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)  # 给第一层LSTM的输出加了一个dropout

        # 如果 LSTM 的层数大于 1, 则总计创建 num_layers 个 LSTM 层
        # 并将所有的 LSTM 层包装进 MultiRNNCell 这样的序列化层级模型中
        # state_is_tuple=True 表示接受 LSTMStateTuple 形式的输入状态
        if n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)

        # dynamic_rnn（动态 RNN）可以让不同迭代传入的 Batch 可以是长度不同的数据
        # 但同一次迭代中一个 Batch 内部的所有数据长度仍然是固定的
        # dynamic_rnn 能更好处理 padding（补零）的情况，节约计算资源
        # 返回两个变量：
        # 第一个是一个 Batch 里在时间维度（默认是 35）上展开的所有 LSTM 单元的输出，形状默认为 [20, 35, 650]，之后会经过扁平层处理
        # 第二个是最终的 state（状态），包含 当前时刻 LSTM 的输出 h(t) 和 当前时刻的单元状态 C(t)
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # 扁平化处理，改变输出形状为 (batch_size * num_steps, hidden_size)，从[20,35,650]到[700, 650]
        output = tf.reshape(output, [-1, hidden_size])

        # 经过softmax从650维转为10000维的one_hot编码 [700,650]->[700,10000]
        # softmax的权重
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))

        # softmax的偏置
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))

        # logits 是 Logistic Regression（用于分类）模型（线性方程： y = W * x + b ）计算的结果（分值）
        # logits 会用softmax转成百分比概率
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # [700,1000]->[20,35,10000]
        logits = tf.reshape(logits, [self.batch_size, self.n_steps, vocab_size])

        # 计算logits的序列的交叉熵的损失
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,  # 期望输出，[20,35]
            tf.ones([self.batch_size, self.n_steps], dtype=tf.float32),  # 权重，设为1
            average_across_timesteps=False,  # 不在时间维度上计算损失大小
            average_across_batch=True  # 在batch上算平均的损失大小
        )

        # 计算各个维度上的loss平均值
        self.cost = tf.reduce_mean(loss)

        # 为计算softmax的概率值，变二维
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))

        # 取最大概率的那个值作为预测
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)

        # 预测值和真实值进行对比
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))

        # 将bool值变为float值，然后取平均值
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 如果是测试，则直接退出
        if not is_training:
            return  # return 可退出函数

        # 学习率 trainable=False 表示不可被训练
        self.learning_rate = tf.Variable(0.0, trainable=False)

        # 返回图中所有可被训练的变量，及除learning_rate之外的变量
        trainable_vars = tf.trainable_variables()

        # 生成一个梯度下降优化器
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # tf.clip_by_global_norm 实现梯度裁剪 gradient clipping为了防止梯度爆炸
        # tf.gradients 计算self.cost对于 trainable_vars的梯度，返回一个梯度列表
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars), 5)

        # apply_gradients将之前裁剪过的梯度 应用到可被训练的变量上去，做梯度下降
        # minimize方法 包含两步 1.计算梯度 2.应用梯度到可训练的变量中去
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                                                  global_step=tf.train.get_or_create_global_step())
        # 用于更新学习率
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
