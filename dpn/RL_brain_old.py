import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class DeepQNetwork:

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greed_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greed_increment
        self.epsilon = 0 if e_greed_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        # n_features * 2 + 2  这里是如何计算的，s, a, r, s_，其中 s + s_ = n_features * 2， a, r = 2
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2)))

        self._build_net()
        self.sess = tf.compat.v1.Session()

        if output_graph:  # 是否输出tensorboard文件
            tf.compat.v1.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # 记录误差
        self.cost_his = []

    def _build_net(self):
        # -------------------------- eval net Q估计 ------------------------
        # 神经网络输入的state信息，通过神经网络可以计算出Q估计
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        # q_target 是 q现实
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.compat.v1.variable_scope('eval_net'):  # 「网络参数的共享，和tf.get_variable()一起使用」
            c_name, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], \
                10, \
                tf.random_normal_initializer(0, 0.3), \
                tf.constant_initializer(0.1)

            # n_l1为隐藏层的大小
            with tf.compat.v1.variable_scope("l1"):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_name)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.compat.v1.variable_scope("l2"):
                w2 = tf.compat.v1.get_variable('w2', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_name)
                b2 = tf.compat.v1.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_name)
                # 返回网络
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.compat.v1.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.compat.v1.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.name_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------------- target net, q真实 ------------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.compat.v1.placeholder.variable_scope('target_net'):  # 「网络参数的共享，和tf.get_variable()一起使用」
            c_name = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            # n_l1为隐藏层的大小
            with tf.compat.v1.variable_scope("l1"):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_name)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.compat.v1.variable_scope("l2"):
                w2 = tf.compat.v1.get_variable('w2', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_name)
                b2 = tf.compat.v1.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_name)
                self.q_target = tf.matmul(l1, w2) + b2

        # target net不需要训练

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 创建一条数据
        transition = np.hstack((s, [a, r], s_))

        # 找到数据需要插入的位置
        index = self.memory_counter % self.memory_size
        # 将数据插入到对应的位置
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # 输入的observation是一个一维数组，需要转换成二维数组
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # self.q_eval是一个网络，feed_dict是输入的数据，返回结果是定义的q_eval
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        # self.memory.sample 和 self.memory.iloc[:self.memory_counter, :].sample 可能memory里有数据不是新生成的，而是之前生成的
        # 随机抽样
        batch_memory = self.memory.sample(self.batch_size) if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter, :].sample(self.batch_size)

        # 拿出一批数据，这里同时执行了两个网络，q_next是真实的值，q_eval是预测的值
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval], feed_dict={
            # 这里和memory的存放方式有关：s, a, r, s_，当前的observation是前n_features个的数据，后observation是后n_features个的数据
            self.s_: batch_memory.iloc[:, -self.n_features:],  # next observation
            self.s: batch_memory.iloc[:, :self.n_features],  # observation
        })

        # 反向传递是根据q估计得值，但是q_eval和q_target选择最终的action可能不一样，需要做归一化。q_eval返回的是一个向量
        # 计算逻辑如下：
        # 1. 先将q_target 等于 q_eval
        # 2. 再将memory中选择出的action在q_target中覆盖掉，这样再做q_eval - q_target时，只有选择到的action对应的值是有结果的
        q_target = q_eval.copy()
        # batch_index行数列表
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 评估q对弈的action，这里是真实结果，从memory里获取到的
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 评估q对弈的reward，这里是真实结果
        reward = batch_memory[:, self.n_features + 1]
        # np.max(q_next, axis=1) 找一个每行最大的结果
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 这里只是计算了真实结果的q_target，评估的target是在self._train_op和self.loss中计算的的，也会传入state状态

        # 训练网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory.iloc[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()









