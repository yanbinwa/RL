import numpy as np
import pandas as pd


class RL(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 贪心加随机
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


class SarsaLambdaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn(self, s, a, r, s_, a_):
        # 先更新eligibility_trace，并且计算trace_decay
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # sarsa lambda 需要更新当前的qtable和之前到达当前状态的所有状态的qtable
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r

        error = q_target - q_predict

        # method 1
        # self.eligibility_trace.ix[s, a] += 1

        # method 2
        self.eligibility_trace.ix[s, :] *= 0
        self.eligibility_trace.ix[s, a] = 1

        self.q_table.ix[s, a] += self.lr * error * self.eligibility_trace.ix[s, a]
        self.eligibility_trace *= self.gamma * self.lambda_

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.eligibility_trace.columns,
                    name=state,
                )
            )


