import numpy as np


def save_data(qnet, lossarr, prob, learn_data, total_steps, episodes):
    avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network(episodes)
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                       np.mean(lossarr[-100]), prob])
    qnet.save()
    np.save('learning_data', learn_data)
