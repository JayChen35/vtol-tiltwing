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


# Environment parameters
ENV_NAME = "LunarLanderContinuous-v2"
ACT_LOW_BOUND         = -1  # Low bound of the action space
ACT_HIGH_BOUND        = 1
NUM_EPISODES          = 750
PPO_STEPS             = 4000 # 500  # Episode length/timestep
RENDER                = False
EVAL_WHILE_TRAIN      = True
EVAL_EPISODE_INTERVAL = 5
SAVE_MODEL            = False
SAVE_INTERVAL         = 10

# Hyperparameters
LR               = 1e-4
GAMMA            = 0.99  # Discount factor
GAE_LAMBDA       = 0.95  # GAE smoothing factor
MINI_BATCH_SIZE  = 20
CLIP_EPSILON     = 0.2  # Clippling parameter
PPO_EPOCHS       = 80  # How much to train on a single batch of experience (PPO is on-policy)
REWARD_THRESHOLD = 90
CRITIC_DISCOUNT  = 0.5  # 0.5  # c1 in the paper (Value Function Coefficient)
ENTROPY_BETA     = 0.05 # 0.01  # c2 in the paper
STD_INITIAL      = 0.5  # Wide initial standard deviation to help exploration
STD_DECAY        = 0.95
STD_MINIMUM      = 0.05


class ActorCritic(tf.keras.Model):
    def __init__(self, obs_space: int, act_space: int):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space
        # Both actor and critic have the same common network
        self.layer_1 = Dense(128, input_shape=(self.observation_space,), kernel_initializer="he_uniform", activation="relu")
        self.layer_2 = Dense(64, kernel_initializer="he_uniform", activation="relu")
        # The Actor explores the environment and maps states to actions (called the "policy").
        # The Critic evalutates the Actor's policy and runs stochastic gradient descent on states->rewards.
        self.actor = Dense(self.action_space, kernel_initializer="glorot_uniform", activation="tanh")
        # The output of the Critic is a single value (the output of the Value Function).
        self.critic = Dense(1, kernel_initializer="glorot_uniform", activation=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.std = STD_INITIAL
        self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(inputs)
        return self.actor(x), tf.squeeze(self.critic(x))

    def update_sigma(self, std_decay=STD_DECAY):
        # Update the standard deviation (random exploration) of the policy. Should be called at the end of an episode
        if self.std > STD_MINIMUM:
            self.std = self.std * std_decay
            self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std


class Trajectory:
    def __init__(self, state: np.ndarray, action: np.ndarray, log_prob: tf.Tensor, reward: float, advantage: tf.Tensor):
        super().__init__()
        # As defined in the paper, one "snapshot" of the environment and policy parameters is called a "Trajectory"
        self.state = state
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.advantage = advantage

    def to_tuple(self):
        return self.state, self.action, self.log_prob, self.reward, self.advantage


@tf.function
def train_step(model: ActorCritic, trajectory: Trajectory):
    # Perform SGD on both the Actor and Critic with the PPO custom loss function
    with tf.GradientTape() as tape:
        tape.watch(model.layer_1.variables)
        tape.watch(model.layer_2.variables)
        tape.watch(model.actor.variables)
        tape.watch(model.critic.variables)
        loss, actor_loss, critic_loss, entropy = ppo_loss(model, trajectory)
    gradients = tape.gradient(loss, model.trainable_variables)
    # print(type(gradients), len(gradients), None in gradients)
    gradients = [grad if grad is not None else tf.zeros_like(var) for (grad, var) in zip(gradients, model.trainable_variables)]
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, actor_loss, critic_loss, entropy  # Don't need to return loss, but returning to keep track of statistics


def run_actor_critic(model: ActorCritic, state: np.ndarray) -> Tuple[tfp.distributions.TruncatedNormal, float]:
    # Simply a wrapper to return the intended outputs for the Actor and Critic (Actor can't return a Distribution)
    mu, value_estimate = model(state)
    # print(mu)
    dist = tfp.distributions.TruncatedNormal(mu, model.sigma, ACT_LOW_BOUND, ACT_HIGH_BOUND)
    return dist, value_estimate


def compute_gae(next_value: tf.Tensor, rewards: list, masks: list, values: list) -> list:
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
        returns.insert(0, gae_value + values[step])
    return returns


def ppo_loss(model: ActorCritic, trajectory: Trajectory):
    state, action, old_log_probs, actual_return, advantage = trajectory.to_tuple()
    # Actual_return = the computed GAE
    dist, value = run_actor_critic(model, state)


    new_log_probs = dist.log_prob(action)
    # Policy_ratio = r(theta) in the paper, basically how much more likely are we to take an action with this new policy?
    policy_ratio = tf.math.exp(new_log_probs-old_log_probs)  # Expressed as a difference of logs
    surr_1 = policy_ratio * advantage
    surr_2 = tf.clip_by_value(policy_ratio, 1.0-CLIP_EPSILON, 1.0+CLIP_EPSILON) * advantage
    # Actor loss follows the clipped objective, while Critic loss is just MSE between returns and the Critic prediction
    actor_loss = -1 * tf.reduce_mean(tf.minimum(surr_1, surr_2))
    critic_loss = tf.reduce_mean(tf.pow(actual_return-value, 2))

    # entropy = tf.reduce_mean(dist.entropy())  # reduce_mean() is just the mean
    entropy = tf.reduce_mean(-(tf.math.exp(new_log_probs) * new_log_probs))

    total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
    return total_loss, actor_loss, critic_loss, entropy


def ppo_minibatch_iterator(states: list, actions: list, log_probs: list, returns: list, advantages: list):
    # Returns a Generator object (volatile Iterator) of size MINI_BATCH_SIZE to update pi (the policy)
    assert len(states) == PPO_STEPS
    # Normalize the advantages and returns to have more reasonable losses (prevent NaN)
    # NOTE: Uncomment this if it doesn't work
    # advantages = tf.linalg.normalize(advantages)[0]
    # returns = tf.linalg.normalize(returns)[0]
    for _ in range(PPO_STEPS // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, PPO_STEPS, MINI_BATCH_SIZE)
        for i in rand_ids:
            trajectory = Trajectory(states[i], actions[i], log_probs[i], returns[i], advantages[i])
            yield trajectory
            

def ppo_update(model: ActorCritic, states: list, actions: list, log_probs: list, returns: list, advantages: list):
    loss_totals = []
    actor_loss_totals = []
    critic_loss_totals = []
    entropies = []
    # Epochs = the number of times we go through the ENTIRE batch of trajectories
    for epoch in range(PPO_EPOCHS):
        # Grabs random mini-batches of trajectories several times until all the data is covered
        for trajectory in ppo_minibatch_iterator(states, actions, log_probs, returns, advantages):
            total_loss, actor_loss, critic_loss, entropy = train_step(model, trajectory)
            loss_totals.append(total_loss)
            actor_loss_totals.append(actor_loss)
            critic_loss_totals.append(critic_loss)
            entropies.append(entropy)
    return np.mean(loss_totals), np.mean(actor_loss_totals), np.mean(critic_loss_totals), np.mean(entropies)


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
    kb.set_floatx("float64")  # Set float data type standard
    tf.config.run_functions_eagerly(True)  # Running eagerly, prevents @tf.function error
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    model = ActorCritic(observation_space, action_space)

    if SAVE_MODEL:
        tensorboard_path, saved_model_path, checkpoint_path = form_results()

    test_rewards = []
    early_stop = False
    episode = 0
    num_games, running_reward, running_length = 0, 0, 0
    LOG_INTERVAL = 1
    while episode < NUM_EPISODES and not early_stop:
        states    = []
        actions   = []
        log_probs = []
        rewards   = []
        masks     = []
        values    = []
        state = np.reshape(env.reset(), [1, observation_space])  # To feed into the network
        total_reward = 0
        for step in range(PPO_STEPS):
            # This loop collects experiences gathered by the agent (Trajectories)
            if RENDER:
                env.render()
            dist, value = run_actor_critic(model, state)
            action = dist.sample()[0]
            next_state, reward, terminal, _ = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            total_reward += reward
            log_prob = dist.log_prob(action)  # tf.Tensor

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(0 if terminal else 1)  # Append the opposite of terminal
            states.append(state)
            actions.append(action)
            state = next_state

            running_reward += reward
            running_length += 1
            if terminal:
                state = np.reshape(env.reset(), [1, observation_space])  # Reset the environment
                num_games += 1
                if num_games % LOG_INTERVAL == 0:
                    print("Game: {}, Avg reward: {}, Avg length: {}".format(num_games, running_reward // LOG_INTERVAL, running_length // LOG_INTERVAL))
                    running_reward = 0
                    running_length = 0

        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)
        advantages = [tf.math.subtract(returns[x], values[x]) for x in range(len(returns))]
        # advantages = tf.linalg.normalize(advantages)[0]
        avg_total_loss, avg_actor_loss, avg_critic_loss, avg_entropy = ppo_update(model, states, actions, log_probs, returns, advantages)
        print("[Episode {}] Total reward: {}. AVG total loss: {}. AVG Actor loss: {}. AVG Critic loss: {}. AVG entropy: {}. STD: {}.".format(
            episode, total_reward, avg_total_loss, avg_actor_loss, avg_critic_loss, avg_entropy, model.std))
        episode += 1
        model.update_sigma()  # Decay the STD every episode
        # Quickly test if the network has reached the threshold reward
        test_rewards = [-float("inf")]
        if EVAL_WHILE_TRAIN and episode % EVAL_EPISODE_INTERVAL == 0:
            test_reward = np.mean([eval_current_policy(model) for _ in range(5)])
            if test_reward >= REWARD_THRESHOLD: early_stop = True
            if test_reward > max(test_rewards) and SAVE_MODEL:
                model.save(saved_model_path + '/Model_eval', overwrite=True)
                print("Models saved on Episode {} since reward > previous average reward ({}>{}).".format(
                    episode, test_reward, max(test_rewards)))
            else:
                print("Models NOT saved since the current reward ({}) is not greater than the max reward ({}).".format(
                    test_reward, max(test_rewards)))
            test_rewards.append(test_reward)
            env.close()
        # Save the model
        if SAVE_MODEL and episode % SAVE_INTERVAL == 0:
            model.save(saved_model_path + '/Model_interval', overwrite=True)
            print("ActorCritic models successfully saved on Episode {}.".format(episode))
