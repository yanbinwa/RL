from RL_brain import SarsaLambdaTable
from maze_env import QLearningEnv


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # 先选择action，再计算reward
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # 这里是基于observation_再选择action
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = QLearningEnv()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
