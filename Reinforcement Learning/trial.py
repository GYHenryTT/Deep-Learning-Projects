import gym
import random
import numpy as np
import tensorflow as tf

from RL_QG_agent import RL_QG_agent
import alpha_beta as ab

env = gym.make('Reversi8x8-v0')

max_epochs = 10
d_size = 200



# Build experience pool
s1 = np.zeros([d_size, 64])
s2 = np.zeros([d_size, 64])
a = np.zeros([d_size, 64])
r = np.zeros([d_size])


agent = RL_QG_agent()
for epoch in range(10):
# for i in range(2):
    w_win = 0
    b_win = 0
    t_sum = 0
    t = 0
    for i_episode in range(max_epochs):
        observation = env.reset()

    # observation is a 3 x 8 x 8 list, representing current game. See more info in state of reversi.py
        for t in range(100):
            action = [1, 2]
            # action contains two int variables, action[0] represents next move, action[1] is chess color

            ################### Black B ############################### 0 means black
            # Black is using alpha-beta search
            # env.render()  # print current chessboard 

            enables = env.possible_actions
            if len(enables) == 0:
                action_ = env.board_size ** 2 + 1
            else:
                if epoch == 0:
                    action_ = agent.place(observation, enables, player=0, version='', greedy=True,
                                          e_greedys=0.1)
                else:
                    action_ = agent.place(observation, enables, player=0, version='2', greedy=True,
                                          e_greedys=0.1)
                # action_ = random.choice(enables)
                # action_ = ab.place(observation, enables, 0)  # 0 means black

            action[0] = action_
            action[1] = 0  # Black B is 0
            observation, reward, done, info = env.step(action)

            if t_sum + t + 1 < d_size:
                s1[t_sum + t + 1, :] = np.reshape(observation[1]-observation[0], [1, -1])
                s2[t_sum + t, :] = np.reshape(observation[1]-observation[0], [1, -1])

            ################### white  W ############################### 1 means white
            # env.render()
            enables = env.possible_actions
            # if nothing to do ,select pass
            if len(enables) == 0:
                action_ = env.board_size ** 2 + 1  # pass
            else:
                if epoch == 0:
                    action_ = agent.place(observation, enables, player=1, version='', greedy=True,
                                          e_greedys=0.1)
                else:
                    action_ = agent.place(observation, enables, player=1, version='2', greedy=True,
                                          e_greedys=0.1)
                # action_ = random.choice(enables)
                # load model

            if action_ not in enables:  # if output is invalid, the opponent wins
                print("Black wins!", enables)
                b_win += 1
                break

            action[0] = action_
            action[1] = 1  # White is 1
            observation, reward, done, info = env.step(action)

            if t_sum + t + 1 < d_size:
                r[t_sum + t] = reward
                if action_ < env.board_size ** 2+1:
                    a[t_sum + t + 1, action_] = 1

            if done:  # Game over
                env.render()
                black_score = len(np.where(env.state[0, :, :] == 1)[0])
                white_score = len(np.where(env.state[1, :, :] == 1)[0])
                t_sum += t+1

                if black_score > white_score:
                    print("Black wins!")
                    b_win += 1

                elif black_score < white_score:
                    print("White wins!")
                    w_win += 1
                else:
                    print("Draw!")
                print(black_score)
                break

            elif t_sum + t + 1 > d_size-1:
                t_sum += t+1
                break

        print("epoch = ", epoch, "  Black：", b_win, "  White：", w_win)

        if t_sum > d_size-1:
            print('t_sum = ', t_sum)
            break

    tf.reset_default_graph()
    # agent.run_training(s1, s2, a, r, d_size, version_load='', version_save='')
    if epoch == 0:
        agent.run_training(s1, s2, a, r, d_size, version_load='', version_save='2')
    else:
        agent.run_training(s1, s2, a, r, d_size, version_load='2', version_save='2')