import tensorflow as tf
import os
import random
import numpy as np


class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")

    def init_model(self, s_crr, s_prv, action_prv, action_crr, reward, enables=None, model='deep sarsa',
                   learning_rate=0.01, gamma=1, greedy=False, e_greedys=0.1
                   ):
        # Greedy Strategy
        def e_greedy(action_max, enables, e):
            a = tf.random_uniform([1, 64], minval=1, maxval=2, dtype=tf.float32)
            if random.random() < 1-e:
                return action_max
            else:
                return tf.argmax(a * enables, 1)

        # def convert_q(q, enables_conv):
        #     a = tf.zeros([1, 64])
        #     for i in enables_conv:
        #         a[1, i] = 1
        #     q_conv = q * a
        #     return q_conv

        end_points = {}

        # DSN Model
        if model == 'deep sarsa':
            # FNN-layer1
            w1 = tf.Variable(tf.random_uniform([64, 100], minval=-0.5, maxval=0.5))
            b1 = tf.Variable(tf.constant(0.1, shape=[1, 100]))
            # FNN-layer2
            w2 = tf.Variable(tf.random_uniform([100, 64], minval=-0.5, maxval=0.5))
            b2 = tf.Variable(tf.constant(0.1, shape=[1, 64]))

            qhidden_crr = tf.nn.sigmoid(tf.matmul(s_crr, w1) + b1)
            q_crr = tf.nn.sigmoid(tf.matmul(qhidden_crr, w2) + b2)

            if s_prv is not None:  # Training mode
                qhidden_prv = tf.nn.sigmoid(tf.matmul(s_prv, w1) + b1)
                q_prv = tf.nn.sigmoid(tf.matmul(qhidden_prv, w2) + b2)
                if action_crr is None:
                    y = reward
                else:
                    y = reward + tf.reduce_sum(gamma * q_crr * action_crr, 1)
                loss_func = tf.reduce_sum((y - tf.reduce_sum(q_prv * action_prv, 1)) ** 2)
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
                end_points['train_step'] = train_step
                end_points['loss'] = loss_func
            else:  # Playing Mode
                q_crr_conv = q_crr * enables
                action_max = tf.argmax(q_crr_conv, 1)
                if greedy is True:
                    action = e_greedy(action_max, enables, e_greedys)
                else:
                    action = action_max
                end_points['action'] = action

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        return end_points

    def run_training(self, s1, s2, a, r, d_size, model='deep sarsa', version_load='', version_save='', e_greedys=0.1):
        s_crr = tf.placeholder(tf.float32, [None, 64])
        s_prv = tf.placeholder(tf.float32, [None, 64])
        action_prv = tf.placeholder(tf.float32, [1, 64])
        action_crr = tf.placeholder(tf.float32, [1, 64])
        reward = tf.placeholder(tf.float32, [1])
        end_points = self.init_model(s_crr, s_prv, action_prv, action_crr, reward, model=model, e_greedys=e_greedys)
        max_epoch = int(40)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with self.sess as sess:
            # sess.run(init_op)
            self.load_model(model_name=model, version=version_load)
            for epoch in range(max_epoch):
                i = random.randint(10, d_size-4)
                loss, _ = sess.run([
                                    end_points['loss'],
                                    end_points['train_step']
                                    ], feed_dict={s_crr: np.reshape(s2[i, :], [1, -1]),
                                                  s_prv: np.reshape(s1[i, :], [1, -1]),
                                                  action_prv: np.reshape(a[i], [1, -1]),
                                                  action_crr: np.reshape(a[i+1], [1, -1]),
                                                  reward: np.reshape(r[i], [-1])})

                # if epoch % 5 == 0:
                print('epoch =', epoch, 'loss = ', loss)
                # if loss < 0.00001:
                #     print("s_prv:", np.reshape(s1[i, :], [1, 192]), 'action_prv:', np.reshape(a[i], [1, -1]),
                #           'action_crr:', np.reshape(a[i+1], [1, -1]),'reward: ',np.reshape(r[i], [-1]))
            self.save_model(model_name=model, version=version_save)

    def place(self, state, enables, player, model='deep sarsa', version='', greedy=False, e_greedys=0.1):
        # This is a test function. Return action from 0 to 63
        # action is next move in Reversi
        tf.reset_default_graph()
        state = state[1]-state[0]
        if player == 0:
            state = state * (-1)
        s_crr = tf.placeholder(tf.float32, [None, 64])
        enables_tf = tf.placeholder(tf.float32, [None, 64])
        end_points = self.init_model(s_crr=s_crr, s_prv=None, action_prv=None, action_crr=None, reward=None,
                                     enables=enables_tf, model=model, greedy=greedy, e_greedys=e_greedys)
        try:
            enables_conv = np.zeros([1, 64])
            for i in enables:
                enables_conv[0, i] = 1

            with self.sess as sess:
                self.load_model(model_name=model, version=version)
                action = sess.run(end_points['action'], feed_dict={s_crr: np.reshape(state, [1, -1]),
                                                                   enables_tf: enables_conv})
        except IndexError:
            action = enables[0]

        return action

    def save_model(self, model_name, version):  # Save model
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter'+model_name+version+'.ckpt'))

    def load_model(self, model_name, version):     # Reload model
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter'+model_name+version+'.ckpt'))



