import tensorflow as tf
from deeprl.helpers.image_processing import preprocess
import numpy as np
import random
from deeprl.environments.atari_env import AtariEnvironment
from deeprl.networks.convolutional_network import Convnet


class TDNet:

    def __init__(self, relations, credits, rom='/home/john/code/pythonfiles/my_dqn/Breakout.bin',
                 save_path="/saved_models/TD.ckpt", load_path=""):

        self.save_path = save_path
        self.load_path = load_path

        self.rom = rom
        self.sess = tf.Session()
        self.env = AtariEnvironment(self.rom)
        self.relations = tf.constant(relations, tf.float32)
        self.credits = credits
        self.dim = len(relations)
        self.nodes, self.node_targets = self.initialize_question_network()
        self.answer_network = self.initialize_answer_network()
        self.loss, self.train_operation, self.credit_vector = self.initialize_learning_procedure()

        self.replay_memory = []

        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        self.load()

    def save(self):
        self.saver.save(self.sess, save_path=self.save_path)

    def load(self):
        if self.load_path:
            self.saver.restore(self.sess, self.load_path)
            print("Model at path ", self.load_path, " loaded.")
        else:
            print("No model provided.")

    def initialize_question_network(self):
        nodes = tf.placeholder(tf.float32, shape=[None, self.dim])
        node_targets = tf.matmul(nodes, self.relations)
        return nodes, node_targets

    def initialize_answer_network(self):
        answer_network = Convnet(self.dim, self.sess)
        return answer_network

    def initialize_learning_procedure(self):
        credit_vector = tf.placeholder(tf.float32, shape=[self.dim, 1])
        action_readout = tf.matmul(self.node_targets, credit_vector)
        loss = tf.reduce_mean(tf.square(tf.sub(action_readout, self.answer_network.output)))
        optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        gradients_and_vars = optimizer.compute_gradients(loss)
        gradients = [gravar[0] for gravar in gradients_and_vars]
        vars = [gravar[1] for gravar in gradients_and_vars]
        clipped_gradients = tf.clip_by_global_norm(gradients, 1.)[0]
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, vars))
        return loss, train_operation, credit_vector

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)

    def true_step(self, prob, state, obs2, obs3, obs4, env):
        output = self.sess.run(self.answer_network.output, feed_dict={self.answer_network.input: [np.array(state)]})
        Q_vals = output[0][-len(self.env.action_space):]
        if random.uniform(0, 1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, Q_vals.max(), step_done

    def learning_step(self, target_weights, batch_size):
            minibatch = random.sample(self.replay_memory, batch_size)
            next_states = np.array([m[3] for m in minibatch])
            feed_dict = {self.answer_network.input: next_states}
            feed_dict.update(zip(self.answer_network.weights, target_weights))
            predictions = self.sess.run(self.answer_network.output, feed_dict=feed_dict)
            targets = self.sess.run(self.node_targets, feed_dict={self.nodes: predictions})
            action_indexes = [m[1] for m in minibatch]
            states = [m[0] for m in minibatch]
            feed_dict = {self.answer_network.input: np.array(states), self.node_targets: targets,
                         self.credit_vector: [credits[:, index] for index in action_indexes]}
            _, loss_val = self.sess.run(fetches=(self.train_operation, self.loss), feed_dict=feed_dict)

            return loss_val

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
            # done = False
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







