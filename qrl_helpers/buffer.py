# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import operator


class Buffer:
    """ Data holder for Quantum Policy Gradient algorithm """
    def __init__(self, batch_size, gamma=1.0):
        assert 0 <= gamma <= 1, 'The discount factor `gamma` has t be between in [0.0, 1.0].'
        self.batch_size = batch_size
        self.gamma = gamma
        self.total_steps = 0
        self.running_rewards = {}
        self.epoch_counter = 0
        self.epoch_rewards = None
        self.terminated, self.truncated = None, None
        self.observations, self.actions, self.rewards = None, None, None

    def reset(self):
        self.epoch_rewards = None
        self.terminated = np.array([False for _ in range(self.batch_size)])
        self.truncated = np.array([False for _ in range(self.batch_size)])
        self.observations = [[] for _ in range(self.batch_size)]
        self.actions = [[] for _ in range(self.batch_size)]
        self.rewards = [[] for _ in range(self.batch_size)]

    def append_observations(self, observations):
        assert self.observations is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, observation in zip(self._not_done, observations):
            self.observations[batch_index].append(observation)

    def append_actions(self, actions):
        assert self.actions is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, action in zip(self._not_done, actions):
            self.actions[batch_index].append(action)
        # construct 0-padded actions to account for already terminated environments
        actions_padded = np.zeros((self.batch_size,), dtype=int)
        actions_padded[self._not_done] = actions
        return actions_padded

    def append_rewards(self, rewards):
        assert self.rewards is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, reward in zip(self._not_done, rewards):
            self.rewards[batch_index].append(reward)

    def update_terminated_and_truncated(self, terminated, truncated):
        # select those that were not terminated/truncated in the previous step
        indices_not_done = self._not_done
        terminated, truncated = terminated[indices_not_done], truncated[indices_not_done]
        self.terminated[indices_not_done], self.truncated[indices_not_done] = terminated, truncated
        # return the indices of the environments that are still not done (i.e. terminated or truncated)
        return self._not_done

    def finish_epoch(self):
        assert self.observations is not None and self.actions is not None and self.rewards is not None, 'Forgot to call `reset(batch_size)`.'
        self.observations = self._flatten_list(self.observations)
        self.actions = self._flatten_list(self.actions)

        # store for logging
        self.epoch_rewards = [sum(rewards) for rewards in self.rewards]
        self.running_rewards[self.epoch_counter] = self.epoch_rewards
        # self.running_rewards[self.epoch_counter] = np.average(self.epoch_rewards)
        # discount rewards
        self.rewards = self._flatten_list([self._discount_and_cumulate_rewards(rewards) for rewards in self.rewards])

        assert len(self.observations) == len(self.actions) == len(self.rewards), 'Inconsistent trajectory.'
        self.total_steps += len(self.observations)
        self.epoch_counter += 1

    @property
    def batched_observations(self):
        assert self.observations is not None, 'Forgot to call `finish_epoch()`.'
        return self.observations

    @property
    def batched_actions(self):
        assert self.actions is not None, 'Forgot to call `finish_epoch()`.'
        return self.actions

    @property
    def batched_rewards(self):
        assert self.rewards is not None, 'Forgot to call `finish_epoch()`.'
        return self.rewards

    @property
    def get_epoch_reward(self):
        assert self.epoch_rewards is not None, 'Forgot to call `finish_epoch()`.'
        return np.average(self.epoch_rewards), np.max(self.epoch_rewards), np.min(self.epoch_rewards)

    @property
    def get_total_steps(self):
        return self.total_steps

    @property
    def get_running_rewards(self):
        return self.running_rewards

    @property
    def _not_done(self):
        """ Return indices of environments thar are not done (terminated or truncated)
        """
        assert self.terminated is not None and self.truncated is not None, 'Forgot to call `reset(batch_size)`.'
        return np.where(list(map(operator.not_, map(operator.or_, self.terminated, self.truncated))))[0]

    @staticmethod
    def _flatten_list(nested_list):
        flat_list = []
        for individual_list in nested_list:
            flat_list += individual_list
        return flat_list

    def _discount_and_cumulate_rewards(self, rewards):
        discounted_cumulated_rewards = []
        running_cumulated_reward = 0
        for t in range(len(rewards) - 1, -1, -1):
            running_cumulated_reward = rewards[t] + self.gamma * running_cumulated_reward
            discounted_cumulated_rewards.append(running_cumulated_reward)
        discounted_rewards = discounted_cumulated_rewards[::-1]
        return discounted_rewards
