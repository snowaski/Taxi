import gym
import numpy as np
env = gym.make("Taxi-v2")

q_table = np.zeros((env.observation_space.n, env.action_space.n))

#exploration rate values
eps = 1
min_eps_val = 0.01
max_eps_val = 1
eps_decay_rate = 0.001

learning_rate = 0.75
discount_rate = 0.95

max_steps = 100
num_episodes = 10000

#total rewards over all episodes
rewards = []

#episode
for episode in range(num_episodes):
    state = env.reset()
    done = False
    current_reward = 0

    #timestamp
    for _ in range(max_steps):
        #Determines whether the agent explores or exploits
        if np.random.random_sample() > eps:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        #calculates new q value based on the weigted sum of our old value and the learned value
        new_q = (1-learning_rate)*q_table[state, action] + learning_rate*(reward + discount_rate*np.max(q_table[new_state, :]))
        q_table[state, action] = new_q

        state = new_state
        current_reward += reward

        if done == True:
            break
    rewards.append(current_reward)

    #decays the epsilon
    eps = min_eps_val + (max_eps_val - min_eps_val) * np.exp(-eps_decay_rate*episode)

#displays a taxi game with final q values
state = env.reset()
total_rewards = 0
for _ in range(max_steps):
    env.render()
    action = np.argmax(q_table[state, :])

    new_state, reward, done, info = env.step(action)

    state = new_state
    total_rewards += reward

    if done:
        print("Score: ", total_rewards)
        break


#finds the reward per thousand episodes
print("\nAverage reward per 1000 episodes: ")
rewards_per_thousand = np.split(np.array(rewards), num_episodes/1000)
count = 1000
for r in rewards_per_thousand:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

env.close()
