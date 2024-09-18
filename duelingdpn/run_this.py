from maze_env import QLearningEnv
from RL_brain_old import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        while True:
            # refresh env
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            # 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            if step > 200 and step % 5 == 0:
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    # end of game
    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = QLearningEnv()
    RL = DeepQNetwork(actions=list(range(env.n_actions)))
    run_maze()
    env.mainloop()
