import sys

import gymnasium as gym
import torch

from core.standard_network import FFNetwork
from core.snn_network_small import SNN_small
from torch.distributions import Categorical
import numpy as np
import core.snn_utils as SNN_utils


class Eval_module:
    def __init__(self, env, actor_state, NType):

        self.env = env

        self.actor_model = actor_state

        print(f"Testing {self.actor_model}", flush=True)

        # If the actor model is not specified, then exit
        if self.actor_model == '':
            print(f"Didn't specify model file. Exiting.", flush=True)
            sys.exit(0)

        # Determine if the action space is continuous or discrete
        if isinstance(env.action_space, gym.spaces.Box):
            print("Using a continuous action space")
            self.continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            print("Using a discrete action space")
            self.continuous = False
        else:
            raise NotImplementedError("The action space type is not supported.")

        # Extract out dimensions of observation and action spaces
        obs_dim = self.env.observation_space.shape[0]

        if self.continuous:
            self.act_dim = self.env.action_space.shape[0]
        else:
            act_dim = self.env.action_space.n

        print("obs dim :", obs_dim)
        print("act dim :", act_dim)

        # Build our policy the same way we build our actor model in PPO
        # policy = ActorNetwork(obs_dim, act_dim)
        if NType == "ANN":
            self.actor = FFNetwork(obs_dim, act_dim)
        else:
            self.actor = SNN_small(obs_dim, act_dim, num_steps=32, add_weight=0.1)

        # Load in the actor model saved by the PPO algorithm
        self.actor.load_state_dict(torch.load(self.actor_model))

    def eval_policy_ANN(self, n_eval_episodes=5):
        """
        Evaluates the given actor (policy) in the environment for a fixed number of episodes.

        :param n_eval_episodes: Number of episodes to evaluate over.
        :return: Mean reward over all episodes.
        """
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset(options={})
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # Get action from the actor (policy network)

                logits = self.actor.forward(obs)

                dist = Categorical(logits=logits)
                action = dist.sample().detach().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print("reward for this episode :", episode_reward)

        # Calculate mean reward over all evaluation episodes
        mean_reward = np.mean(rewards)
        return mean_reward

    def eval_policy_SNN(self, n_eval_episodes=5, num_steps=32, threshold=None, shift=None):
        """
        Evaluates the given actor (policy) in the environment for a fixed number of episodes.

        :param n_eval_episodes: Number of episodes to evaluate over.
        :return: Mean reward over all episodes.
        """
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset(options={})
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # Get action from the actor (policy network)

                obs_st = SNN_utils.generate_spike_trains(obs,
                                                         num_steps=num_steps,
                                                         threshold=threshold,
                                                         shift=shift)

                logits, _ = self.actor.forward(obs_st)  # Assuming 'forward' method in actor handles the action logic

                dist = Categorical(logits=logits)
                action = dist.sample().detach().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print("reward for this episode :", episode_reward)

        # Calculate mean reward over all evaluation episodes
        mean_reward = np.mean(rewards)
        return mean_reward