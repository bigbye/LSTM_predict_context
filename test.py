from utills import *
from network import *


def test(model_path, test_data, vocab_size, id_to_word):
    # 测试的输入
    test_input = Input(batch_size=20, n_steps=35, data=test_data)

    # 创建测试的模型，基本的超参数需要和训练时用的一致，例如：
    # hidden_size，num_steps，num_layers，vocab_size，batch_size 等等
    # 因为我们要载入训练时保存的参数的文件，如果超参数不匹配 TensorFlow 会报错
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, n_layers=2)

    # Saver来恢复训练时产生的模型的变量
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        saver.restore(sess, model_path)

        # 测试30个批次
        num_acc_batches = 30

        # 打印预测单词和实际单词的批次数
        check_batch_idx = 25

        # 超过5个批次才开始累加精度
        acc_check_thresh = 5

        # 初试精度的和，用于之后算平均精度
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                          feed_dict={m.init_state: current_state})
                pred_words = [id_to_word[x] for x in pred[:m.n_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print(" ".join(true_words))
                print("预测的单词：")
                print(" ".join(pred_words))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc

        print('平均精度：{:.3f}'.format(accuracy / num_acc_batches - acc_check_thresh))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    trained_model = save_path + os.sep + load_file

    test(trained_model, test_data, vocab_size, id_to_word)
