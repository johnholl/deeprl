from deeprl.agents.td_network import TDNet
from deeprl.helpers.image_processing import preprocess
from deeprl.helpers.checkpointing import save_data
from deeprl.helpers.td_matrix_generators import DQN_TD_matrix
import numpy as np
from threading import Thread
from optparse import OptionParser
import random


# parse command line options
parser = OptionParser()
parser.add_option("-b", "--batch-size", dest="BATCH_SIZE", help="mini-batch size for training", default=32,
                  type=int, metavar="BATCH SIZE")

parser.add_option("-s", "--steps", dest="STEPS", help="number of total steps", default=20000000,
                  type=int, metavar="STEPS")

parser.add_option("-r", "--rom", dest="ROM", help="rom file location", default="../environments/Breakout.bin", metavar="ROM")

parser.add_option("-l", "--logfile", dest="SAVE_FILE", help="model checkpoint location",
                   default="../saved_models/model.ckpt", metavar="FILE")

parser.add_option("-t", "--training-length", dest="TRAIN_EPISODES", help="Number of episodes used for testing agent",
                   default=20, type=int, metavar="TRAIN EPISODES")


(options, args) = parser.parse_args()
BATCH_SIZE = options.BATCH_SIZE
STEPS = options.STEPS
ROM = options.ROM
TRAIN_EPISODES = options.TRAIN_EPISODES

relations, credit = DQN_TD_matrix(4)

dqn = TDNet(relations, credit, 1, rom=ROM)

target_weights = dqn.sess.run(dqn.answer_network.weights)
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
episode_number = 0

while total_steps < 20000000:
    obs1 = dqn.env.reset()
    obs2 = dqn.env.step(dqn.env.sample_action())[0]
    obs3 = dqn.env.step(dqn.env.sample_action())[0]
    obs4, _, terminal = dqn.env.step(dqn.env.sample_action())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0

    while not dqn.env.ale.game_over():
        prob, action, reward, new_state, obs1, obs2, obs3, obs4, _, terminal = dqn.true_step(prob, state, obs2, obs3, obs4, dqn.env)
        dqn.update_replay_memory((state, action, reward, new_state, terminal))
        state = new_state

        if len(dqn.replay_memory) >= 1000 and total_steps % 4 == 0:
            minibatch = random.sample(dqn.replay_memory, BATCH_SIZE)
            expanded_minibatch = []
            for i in range(len(minibatch)):
                minibatch[i] = minibatch[i] + ([minibatch[i][2]],)

            loss_val = dqn.learning_step(target_weights, minibatch)
            loss_vals.append(loss_val)

        if total_steps % 10000 == 0:
            target_weights = dqn.sess.run(dqn.answer_network.weights)

        if total_steps % 50000 == 0:
            testing_thread = Thread(target=save_data, args=(dqn, loss_vals, prob,
                                                            learning_data, total_steps, TRAIN_EPISODES))
            testing_thread.start()

        total_steps += 1
        steps += 1

    episode_number += 1

    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
