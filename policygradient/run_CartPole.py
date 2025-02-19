"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

'''

import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)


if __name__ == "__main__":
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

'''
import gymnasium as gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v1', render_mode="human")
# env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

if __name__ == "__main__":

    for i_episode in range(3000):

        observation, info = env.reset(seed=42)

        while True:
            if RENDER:
                env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, _, info = env.step(action)

            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True     # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()

                if i_episode == 0:
                    plt.plot(vt)    # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            observation = observation_
