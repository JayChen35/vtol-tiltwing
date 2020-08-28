# Proximal Policy Optimization - ReachAndAvoid-v0
# Jason Chen & Srikar Gouru
# NMLO team name: GMD
# 17 August, 2020

"""
Resources: 
Explaination video of PPO: https://www.youtube.com/watch?v=5P7I-xPq8u8
Explaination article: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
Actor/Critic PPO: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
Clean PyTorch implementation: https://github.com/JayChen35/RL-Adventure-2/blob/master/3.ppo.ipynb
Getting a single file from Github: $ curl -L -O https://github.com/JayChen35/RL-Adventure-2/raw/master/3.ppo.ipynb
Understanding PPO hyperparameters: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
Kernels: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
Short Pytorch implementation: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
"""

import tensorflow as tf
import tensorflow.keras.backend as kb
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
import numpy as np
import gym
import os
from datetime import datetime
from typing import Tuple
import random
import time


# Environment parameters
ENV_NAME = "LunarLanderContinuous-v2"
ACT_LOW_BOUND         = -1  # Low bound of the action space
ACT_HIGH_BOUND        = 1
NUM_EPISODES          = 750
PPO_STEPS             = 4000 # 500  # Episode length/timestep
RENDER                = False
EVAL_WHILE_TRAIN      = False
EVAL_EPISODE_INTERVAL = 5
SAVE_MODEL            = False
SAVE_INTERVAL         = 10
MAX_EPISODE_LENGTH = 10000

# Hyperparameters
LR               = 1e-4
GAMMA            = 0.99  # Discount factor
GAE_LAMBDA       = 0.95  # GAE smoothing factor
MINI_BATCH_SIZE  = PPO_STEPS # Just a single batch for now
CLIP_EPSILON     = 0.2  # Clippling parameter
PPO_EPOCHS       = 80  # How much to train on a single batch of experience (PPO is on-policy)
REWARD_THRESHOLD = 90
CRITIC_DISCOUNT  = 0.5  # 0.5  # c1 in the paper (Value Function Coefficient)
ENTROPY_BETA     = 0.05 # 0.01  # c2 in the paper
STD_INITIAL      = 0.5  # Wide initial standard deviation to help exploration
STD_DECAY        = 0.95
STD_MINIMUM      = 0.05

def make_tensor(lst: list, dtype=tf.float32):
    if isinstance(lst[0], np.ndarray) or isinstance(lst[0], tf.Tensor):
        if len(lst[0].shape) > 0 and lst[0].shape[0] == 1:
            # Stack them
            return tf.convert_to_tensor(np.concatenate(lst, 0), dtype=dtype)
    return tf.convert_to_tensor(np.array(lst), dtype=dtype)


class ActorCritic(tf.keras.Model):
    def __init__(self, obs_space: int, act_space: int):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space
        # Both actor and critic have the same common network
        self.layer_1 = Dense(64, input_shape=(self.observation_space,), kernel_initializer="he_uniform", activation="relu")
        self.layer_2 = Dense(32, kernel_initializer="he_uniform", activation="relu")
        # The Actor explores the environment and maps states to actions (called the "policy").
        # The Critic evalutates the Actor's policy and runs stochastic gradient descent on states->rewards.
        self.actor = Dense(self.action_space, kernel_initializer="glorot_uniform", activation="tanh")
        # The output of the Critic is a single value (the output of the Value Function).
        self.critic = Dense(1, kernel_initializer="glorot_uniform", activation=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.std = STD_INITIAL
        # self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(inputs)
        return self.actor(x), tf.squeeze(self.critic(x))

    def update_sigma(self, std_decay=STD_DECAY):
        # Update the standard deviation (random exploration) of the policy. Should be called at the end of an episode
        if self.std > STD_MINIMUM:
            self.std = self.std * std_decay
            # self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std


class Trajectory:
    def __init__(self):
        # As defined in the paper, one "snapshot" of the environment and policy parameters is called a "Trajectory"
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []


    def add(self, state: np.ndarray, action: np.ndarray, log_prob: tf.Tensor, reward: float, advantage: tf.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.advantages.append(advantage)
    
    def convert_to_tensors(self):
        self.states = make_tensor(self.states)
        self.actions = make_tensor(self.actions)
        self.log_probs = make_tensor(self.log_probs)
        self.rewards = make_tensor(self.rewards)
        self.advantages = make_tensor(self.advantages)
    
    def to_tuple(self):
        return self.states, self.actions, self.log_probs, self.rewards, self.advantages


class Memory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.masks = []
        self.values = []

    def add_memory(self, state, action, log_prob, reward, mask, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.values.append(value)
    
    def convert_to_tensors(self):
        self.states = make_tensor(self.states)
        self.actions = make_tensor(self.actions)
        self.log_probs = make_tensor(self.log_probs)
        self.rewards = make_tensor(self.rewards)
        self.masks = make_tensor(self.masks)
        self.values = make_tensor(self.values)
    
    def get_minibatches(self, minibatch_size, returns, advantages):
        assert(len(self.states) % minibatch_size == 0)
        length = len(self.states)
        idxs = [i for i in range(length)]
        random.shuffle(idxs)
        batches = []
        for start_idx in range(0, length, minibatch_size):
            batch_idxs = np.array(idxs[start_idx:start_idx + minibatch_size]).reshape(-1, 1)
            trajectory = Trajectory()
            trajectory.states = tf.gather_nd(self.states, batch_idxs)
            trajectory.actions = tf.gather_nd(self.actions, batch_idxs)
            trajectory.log_probs = tf.gather_nd(self.log_probs, batch_idxs)
            trajectory.returns = tf.gather_nd(returns, batch_idxs)
            trajectory.advantages = tf.gather_nd(advantages, batch_idxs)
        return batches


class PPO:
    def __init__(self, model: ActorCritic):
        self.memory = Memory()
        self.model = model


    def get_discounted_rewards(self) -> tf.Tensor:
        discounted = []
        discounted_reward = 0
        for reward, mask in zip(reversed(self.memory.rewards), reversed(self.memory.masks)):
            if not mask:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            discounted.append(discounted_reward)
        return tf.convert_to_tensor(list(reversed(discounted)))


    def compute_gae(self, next_value: tf.Tensor, rewards: list, masks: list, values: list) -> list:
        # GAE = Generalized Advantage Estimation, the "real" return (first term in the advantage function)
        values.append(next_value)
        gae_value = 0
        returns = []
        # Mask (inverse of terminal) is used to prevent us from using a state from the next newly restarted game
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + GAMMA * values[step+1] * masks[step] - values[step]
            gae_value = delta + GAMMA * GAE_LAMBDA * masks[step] * gae_value
            print(gae_value)
            # GAE_lambda is a smoothing parameter to reduce variance in training
            returns.append(gae_value + values[step])
        return list(reversed(returns))


    def update(self, next_state: np.ndarray):
        self.memory.convert_to_tensors()
        _, next_value = self.model(next_state)
        # NOTE: Not sure why I'm doing this, but hopefully it works
        returns = self.get_discounted_rewards()
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-5)
        # returns = tf.linalg.normalize(returns)[0]
        # returns = compute_gae(next_value, rewards, masks, values)
        # assert(returns.shape[0] == len(self.memory.values))

        advantages = returns - self.memory.values

        loss_totals = []
        actor_loss_totals = []
        critic_loss_totals = []
        entropies = []
        # Epochs = the number of times we go through the ENTIRE batch of trajectories
        for epoch in range(PPO_EPOCHS):
            # Grabs random mini-batches of trajectories several times until all the data is covered
            batches = self.memory.get_minibatches(MINI_BATCH_SIZE, returns, advantages)
            for trajectory in batches:
                total_loss, actor_loss, critic_loss, entropy = self.train_trajectory(trajectory)
                loss_totals.append(total_loss)
                actor_loss_totals.append(actor_loss)
                critic_loss_totals.append(critic_loss)
                entropies.append(entropy)
        return np.mean(loss_totals), np.mean(actor_loss_totals), np.mean(critic_loss_totals), np.mean(entropies)


    @tf.function
    def train_trajectory(self, trajectory: Trajectory):
        # Perform SGD on both the Actor and Critic with the PPO custom loss function
        with tf.GradientTape() as tape:
            tape.watch(self.model.layer_1.variables)
            tape.watch(self.model.layer_2.variables)
            tape.watch(self.model.actor.variables)
            tape.watch(self.model.critic.variables)
            loss, actor_loss, critic_loss, entropy = self.ppo_loss(trajectory)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [grad if grad is not None else tf.zeros_like(var) for (grad, var) in zip(gradients, self.model.trainable_variables)]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, actor_loss, critic_loss, entropy  # Don't need to return loss, but returning to keep track of statistics


    def run_actor_critic(self, state: np.ndarray) -> Tuple[tfp.distributions.TruncatedNormal, float]:
        # Simply a wrapper to return the intended outputs for the Actor and Critic (Actor can't return a Distribution)
        mu, value_estimate = self.model(state)
        # Covariance matrix: the diagonal matrix of the variance
        cov_mat = tf.eye(self.model.action_space, dtype=mu.dtype) * self.model.std * self.model.std
        dist = tfp.distributions.MultivariateNormalFullCovariance(mu, cov_mat)
        return dist, value_estimate


    def ppo_loss(self, trajectory: Trajectory):
        states, actions, old_log_probs, returns, advantages = trajectory.to_tuple()
        # Actual_return = the computed GAE
        dist, values = self.run_actor_critic(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        # Policy_ratio = r(theta) in the paper, basically how much more likely are we to take an action with this new policy?
        policy_ratios = tf.math.exp(new_log_probs - old_log_probs)  # Expressed as a difference of logs

        advantages = returns - values
        surr_1 = policy_ratios * advantages
        surr_2 = tf.clip_by_value(policy_ratios, 1.0-CLIP_EPSILON, 1.0+CLIP_EPSILON) * advantages
        # Actor loss follows the clipped objective, while Critic loss is just MSE between returns and the Critic prediction
        actor_loss = -1 * tf.minimum(surr_1, surr_2)
        critic_loss = tf.pow(returns - values, 2)

        # entropy = tf.reduce_mean(dist.entropy())  # reduce_mean() is just the mean
        # entropy = tf.reduce_mean(-(tf.math.exp(new_log_probs) * new_log_probs))

        actor_loss = tf.reduce_mean(actor_loss)
        critic_loss = tf.reduce_mean(critic_loss)
        entropy = tf.reduce_mean(entropy)

        total_loss = actor_loss + CRITIC_DISCOUNT * critic_loss - ENTROPY_BETA * entropy
        actor_loss, critic_loss, total_loss, entropy = tf.reduce_mean(actor_loss), tf.reduce_mean(critic_loss), tf.reduce_mean(total_loss), tf.reduce_mean(entropy)

        return total_loss, actor_loss, critic_loss, entropy
                


def eval_current_policy(model: ActorCritic):
    state = np.reshape(env.reset(), [1, model.observation_space])
    done = False
    total_reward = 0
    step = 0
    while not done and step < PPO_STEPS:
        env.render()
        mu, _ = model(state)
        next_state, reward, done, _  = env.step(mu[0])
        next_state = np.reshape(next_state, [1, observation_space])
        total_reward += reward
        state = next_state
        step += 1
    return total_reward


def form_results():
    folder_name = "./{0}@{1}_ReachAndAvoid-PPO". \
        format(datetime.today().strftime('%Y-%m-%d'), datetime.now().time().strftime('%H%M'))
    tensorboard_path = folder_name + '/Tensorboard'
    saved_model_path = folder_name + '/Saved_models'
    checkpoint_path = folder_name + '/Checkpoints'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(checkpoint_path)
    return tensorboard_path, saved_model_path, checkpoint_path



if __name__ == "__main__":
    LOG_INTERVAL = 1
    kb.set_floatx("float32")  # Set float data type standard
    tf.config.run_functions_eagerly(True)  # Running eagerly, prevents @tf.function error
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    model = ActorCritic(observation_space, action_space)
    ppo = PPO(model)

    if SAVE_MODEL:
        tensorboard_path, saved_model_path, checkpoint_path = form_results()

    test_rewards = []
    early_stop = False
    episode = 0
    time_steps = 0

    total_reward = 0
    total_length = 0
    while episode < NUM_EPISODES and not early_stop:
        state = np.reshape(env.reset(), [1, observation_space])  # To feed into the network
        for t in range(MAX_EPISODE_LENGTH):
            # This loop collects experiences gathered by the agent (Trajectories)
            if RENDER:
                env.render()
            dist, value = ppo.run_actor_critic(state)
            action = dist.sample()[0]
            next_state, reward, terminal, _ = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            log_prob = dist.log_prob(action)  # tf.Tensor

            ppo.memory.add_memory(state, action, log_prob, reward, 0 if terminal else 1, value)
            state = next_state

            if len(ppo.memory.states) >= PPO_STEPS:
                print("Training")
                ppo.update(next_state)
                ppo.memory.clear()
                model.update_sigma()  # Decay the STD every episode

            total_reward += reward
            total_length += 1
            if terminal:
                break


        episode += 1
        if episode % LOG_INTERVAL == 0:
            print("Episode: {}, Avg reward: {}, Avg length: {}".format(episode, total_reward // LOG_INTERVAL, total_length // LOG_INTERVAL))
            total_reward, total_length = 0, 0

        # # Quickly test if the network has reached the threshold reward
        # test_rewards = [-float("inf")]
        # if EVAL_WHILE_TRAIN and episode % EVAL_EPISODE_INTERVAL == 0:
        #     test_reward = np.mean([eval_current_policy(model) for _ in range(5)])
        #     if test_reward >= REWARD_THRESHOLD: early_stop = True
        #     if test_reward > max(test_rewards) and SAVE_MODEL:
        #         model.save(saved_model_path + '/Model_eval', overwrite=True)
        #         print("Models saved on Episode {} since reward > previous average reward ({}>{}).".format(
        #             episode, test_reward, max(test_rewards)))
        #     else:
        #         print("Models NOT saved since the current reward ({}) is not greater than the max reward ({}).".format(
        #             test_reward, max(test_rewards)))
        #     test_rewards.append(test_reward)
        #     env.close()
        # # Save the model
        # if SAVE_MODEL and episode % SAVE_INTERVAL == 0:
        #     model.save(saved_model_path + '/Model_interval', overwrite=True)
        #     print("ActorCritic models successfully saved on Episode {}.".format(episode))
