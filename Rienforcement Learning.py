import gym   # all you have to do to import and use open ai gym!
env = gym.make('FrozenLake-v0')  # we are going to use the FrozenLake enviornment, sets it up
#AI gym is meant for RL, has obs space and action space for every environ, obs tells us how many states in environ
#actions space tells us how many actions at any state
#this game is a frozen lake with holes, so get from start to goal without falling into holes

import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 2500 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96

RENDER = False # if you want to see training set to true

epsilon = 0.9

#Q = np.array [.319233206, .0290153132, .0330154552, 3.87804510e-02], [1.47556526e-03, 1.63561688e-02, 6.55024874e-03, 3.04200719e-01],[6.56838237e-04, 6.44951656e-03, 1.31410435e-02, 1.15346137e-01],[5.74414577e-03, 4.32475561e-03, 1.48688836e-03, 1.17743034e-01], [3.39634885e-01, 1.12859860e-03,2.17608005e-02, 4.41638944e-05], [0.00000000e+00 ,0.00000000e+00, 0.00000000e+00 ,0.00000000e+00], [3.03757178e-07, 1.99514385e-07 ,2.24813448e-01, 2.62485331e-07], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ,0.00000000e+00], [8.03442959e-03, 6.12680978e-03,7.56664105e-03, 6.05357621e-01], [1.29802157e-02, 3.05977153e-01, 4.22037147e-03, 8.94532300e-03], [2.79783088e-01 ,1.11606316e-03, 9.50175590e-04 ,1.91697446e-05], [0.00000000e+00 ,0.00000000e+00 ,0.00000000e+00, 0.00000000e+00], [0.00000000e+00 ,0.00000000e+00 ,0.00000000e+00 ,0.00000000e+00], [9.86727947e-02, 1.51018184e-02, 4.66662528e-01, 2.56638262e-02], [1.99604633e-01 ,7.52911360e-01, 8.03753272e-02, 2.31438593e-01], [0.00000000e+00 ,0.00000000e+00 ,0.00000000e+00, 0.00000000e+00]
rewards = []
for episode in range(EPISODES):

    state = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                    reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")
# and now we can see our Q values!

# we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

