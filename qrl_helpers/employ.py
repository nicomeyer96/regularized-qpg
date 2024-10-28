# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import gymnasium as gym
import torch
from torch.distributions import Categorical
from qiskit_torch_module import QuantumModule

from .buffer import Buffer


def employ_model(
        env: gym.vector.VectorEnv,
        buffer: Buffer,
        qnn: QuantumModule,
) -> tuple[tuple[float, float, float], bool]:
    """ Test model and report reward statistics """

    # reset data buffer (no need to store observations, as we do not perform updates here)
    buffer.reset()
    observations, _ = env.reset()

    while True:

        # store observations (only for consistency, are not used for validation)
        buffer.append_observations(observations)

        with torch.set_grad_enabled(False):
            policies = torch.div(torch.add(torch.mul(qnn(observations).repeat(1, 2), torch.tensor([1.0, -1.0])), 1.0), 2.0)
            # sample action from distribution defined by policy
            dist = Categorical(policies)
            actions = dist.sample()

        actions_padded = buffer.append_actions(actions)
        # Execute action in environment, store reward required for REINFORCE update
        observations_, rewards, terminated, truncated, _ = env.step(actions_padded)

        # store rewards
        buffer.append_rewards(rewards)

        indices_not_done = buffer.update_terminated_and_truncated(terminated, truncated)
        observations = observations_[indices_not_done]

        if 0 == len(indices_not_done):
            buffer.finish_epoch()
            break

    return buffer.get_epoch_reward
