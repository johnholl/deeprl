from deeprl.networks.convolutional_network import Convnet
from deeprl.environments.atari_env import AtariEnvironment
import tensorflow as tf
from deeprl.helpers.image_processing import preprocess
import numpy as np
import random


class DQN:

    def __init__(self, rom='Breakout.bin', save_path="/saved_models/DQN.ckpt", load_path=""):
        self.rom = rom
        self.save_path = save_path
        self.load_path = load_path
        self.env = AtariEnvironment(rom=self.rom)
        self.sess = tf.Session()
        self.OUTPUT_SIZE = len(self.env.action_space)

        self.convnet = Convnet(output_size=self.OUTPUT_SIZE, sess=self.sess)
        self.target = tf.placeholder(tf.float32, None)
        self.action_hot = tf.placeholder('float', [None, self.OUTPUT_SIZE])
        self.action_readout = tf.reduce_sum(tf.mul(self.convnet.output, self.action_hot), reduction_indices=1)
        self.loss = tf.reduce_mean(.5*tf.square(tf.sub(self.action_readout, self.target)))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.vars = [gravar[1] for gravar in self.gradients_and_vars]
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        self.load()

        self.replay_memory = []

    def save(self):
        self.saver.save(self.sess, save_path=self.save_path)

    def load(self):
        if self.load_path:
            self.saver.restore(self.sess, self.load_path)
            print("Model at path ", self.load_path, " loaded.")
        else:
            print("No model provided.")

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)

    def test_network(self):
        test_env = AtariEnvironment(self.rom)
        total_reward = 0.
        total_steps = 0.
        q_avg_total = 0.
        max_reward = 0.
        for ep in range(20):
            obs1 = test_env.reset()
            obs2 = test_env.step(test_env.sample_action())[0]
            obs3 = test_env.step(test_env.sample_action())[0]
            obs4, _, done = test_env.step(test_env.sample_action())
            obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
            state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
            episode_reward = 0.
            num_steps = 0.
            ep_q_total = 0.
            while not test_env.ale.game_over():
                _, action, reward, new_state, obs1, obs2, obs3, obs4, Qval, done =\
                    self.true_step(0.05, state, obs2, obs3, obs4, test_env)
                state = new_state
                episode_reward += reward
                num_steps += 1.
                ep_q_total += Qval
            max_reward = max(episode_reward, max_reward)
            ep_q_avg = ep_q_total/num_steps
            q_avg_total += ep_q_avg
            total_reward += episode_reward
            total_steps += num_steps

        avg_Q = q_avg_total/20.
        avg_reward = total_reward/20.
        avg_steps = total_steps/20.
        print("Average Q-value: {}".format(avg_Q))
        print("Average episode reward: {}".format(avg_reward))
        print("Average number of steps: {}".format(avg_steps))
        print("Max reward over 20 episodes: {}".format(max_reward))

        return avg_Q, avg_reward, max_reward, avg_steps

    # A helper that combines different parts of the step procedure
    def true_step(self, prob, state, obs2, obs3, obs4, env):

        q_vals = self.sess.run(self.convnet.output, feed_dict={self.convnet.input: [np.array(state)]})
        if random.uniform(0,1) > prob:
            step_action = q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, q_vals.max(), step_done

    def learning_step(self, target_weights, batch_size):
            minibatch = random.sample(self.replay_memory, batch_size)
            next_states = np.array([m[3] for m in minibatch])
            feed_dict = {self.convnet.input: next_states}
            feed_dict.update(zip(self.convnet.weights, target_weights))
            q_vals = self.sess.run(self.convnet.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(batch_size)
            action_list = np.zeros((batch_size, self.OUTPUT_SIZE))
            for i in range(batch_size):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] += 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            feed_dict = {self.convnet.input: np.array(states), self.target: target_q, self.action_hot: action_list}
            _, loss_val = self.sess.run(fetches=(self.train_operation, self.loss), feed_dict=feed_dict)

            return loss_val
