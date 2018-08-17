from utills import *
from network import *


def train(train_data, vocab_size, n_layers, n_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_epoch=50):
    # 训练的输入
    training_input = Input(batch_size=batch_size, n_steps=35, data=train_data)

    # 创建训练的模型
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocab_size, n_layers=n_layers)

    # 初始化变量的操作
    init_op = tf.global_variables_initializer()

    # 初试的学习率的衰减率
    orig_decay = lr_decay

    with tf.Session() as sess:
        sess.run(init_op)

        # 协调器用于协调线程的运行
        coord = tf.train.Coordinator()

        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 为了用Saver来保存模型的变量
        saver = tf.train.Saver()  # max_to_keep默认为5

        # 开始epoch的训练
        for epoch in range(n_epochs):
            # 只有 Epoch 数大于 max_lr_epoch（设置为 10）后，才会使学习率衰减
            # 也就是说前 10 个 Epoch 的学习率一直是 1, 之后每个 Epoch 学习率都会衰减
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0)
            m.assign_lr(sess, learning_rate * new_lr_decay)

            # 当前的状态
            # 第二维 是2 是因为对每一个LSTM单元有两个来自上一单元的输入：
            # 一个是 前一时刻 LSTM 的输出 h(t-1)
            # 一个是 前一时刻的单元状态 C(t-1)
            current_state = np.zeros((n_layers, 2, batch_size, m.hidden_size))

            # 获取当前时间，以便打印日志时用
            current_time = datetime.datetime.now()

            for step in range(training_input.step_size):  # 由于
                # train_op 操作：计算被修剪（clipping）过的梯度，并最小化 cost（误差）
                # state 操作：返回时间维度上展开的最后 LSTM 单元的输出（C(t) 和 h(t)），作为下一个 Batch 的输入状态
                if step % print_epoch != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = float((datetime.datetime.now() - current_time).seconds) / print_epoch
                    current_time = datetime.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("epoch{},第{}步，损失:{:.3f},精度:{:.3f},每步所用秒数:{:.2f}".format(epoch, step, cost, acc, seconds))

            saver.save(sess, save_path + os.sep + model_save_name, global_step=epoch)

        saver.save(sess, save_path + os.sep + model_save_name + "-final")

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)
    train(train_data, vocab_size, n_layers=2, n_epochs=70, batch_size=20, model_save_name='train-checkpoint')
