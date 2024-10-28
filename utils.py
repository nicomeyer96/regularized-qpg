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
import argparse
import numpy as np

import warnings
# filter warnings when using CartPole-v0
warnings.filterwarnings('ignore', module='gymnasium', category=DeprecationWarning)
# filter warnings when setting environment states
warnings.filterwarnings('ignore', module='gymnasium', category=UserWarning)


# values to normalize CartPole observations approximately to range [-1, 1]
OBS_NORMALIZER = [2.4, 2.5, 0.21, 2.5]


class CartPoleWrapper(gym.Wrapper):
    """
    CartPole Wrapper with observation perturbation and custom pole initialization
    """

    def __init__(self, env, perturbation: float = 0.0,
                 pole_angle: (float, float) = None, pole_velocity: (float, float) = None):
        """ Initialize custom CartPole environment

            Args:
                env: Environment handle
                perturbation: Standard deviation of gaussian observation perturbation
                pole_angle: Range of pole angles to initialize to (un-normalized)
                pole_velocity: Range of pole angular velocities to initialize to (un-normalized)
        """
        super().__init__(env)
        self.env = env
        self.perturbation = perturbation
        self.pole_angle = pole_angle
        self.pole_velocity = pole_velocity

    def step(self, action):
        """ Perform step, normalize to range [-1, 1] and optionally apply perturbation """
        obs_raw, reward, terminated, truncated, info = self.env.step(action)
        # normalize all values to approx. range [-1, 1]
        obs = obs_raw / OBS_NORMALIZER
        # apply perturbation
        obs += self.perturbation * np.random.randn(4)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """ Reset environment, optionally enforce certain pole initial conditions, normalize to range [-1, 1] """
        obs_raw, info = self.env.reset(**kwargs)
        if self.pole_angle is not None:
            assert self.pole_angle[0] >= -4.8 and self.pole_angle[1] <= 4.8
            pole_angle = np.random.uniform(self.pole_angle[0], self.pole_angle[1])
            # set pole angle (for output ant internal state)
            obs_raw[2], self.state[2] = pole_angle, pole_angle
        if self.pole_velocity is not None:
            pole_velocity = np.random.uniform(self.pole_velocity[0], self.pole_velocity[1])
            # set pole velocity (for output ant internal state)
            obs_raw[3], self.state[3] = pole_velocity, pole_velocity
        # normalize all values to approx. range [-1, 1]
        obs = obs_raw / OBS_NORMALIZER
        # apply perturbation
        obs += self.perturbation * np.random.randn(4)
        return obs, info


def parse():
    parser = argparse.ArgumentParser()
    # For training and testing
    parser.add_argument('--env', type=str, default='CartPole-v0', choices=['CartPole-v0', 'CartPole-v1'])
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount rate')
    parser.add_argument('--layers', type=int, default=3,
                        help='Layers of VQC ansatz')
    parser.add_argument('--seed', type=int, default=None,
                        help='Reproducible parameter initialization')
    parser.add_argument('--episodes', type=int, default=1010,
                        help='Number of episodes to train for')
    parser.add_argument('--batch', type=int, default=10,
                        help='Parallel environments in each epoch')
    parser.add_argument('--reg', type=float, default=0.0,
                        help='Lipschitz regularization weighting')
    parser.add_argument('--val', type=int, default=10,
                        help='Validation frequency for early stopping, set to -1 to disable.')
    parser.add_argument('--threads', type=int, default=0,
                        help='Threads for parallel computation')
    # For testing
    parser.add_argument('--model', type=str,
                        help='[Test] Trained model path (stored in `results/train` folder)')
    parser.add_argument('--runs', type=int, default=100,
                        help='[Test] Environment runs for testing')
    parser.add_argument('--perturbate', action='store_true',
                        help='[Test] Test observation perturbation')
    parser.add_argument('--angle', action='store_true',
                        help='[Test] Test pole angles')
    parser.add_argument('--velocity', action='store_true',
                        help='[Test] Test pole angular velocities')
    # For curriculum training
    parser.add_argument('--failures', type=int, default=1000,
                        help='[Curriculum] Number of allowed failures')
    args = parser.parse_args()
    if args.seed is None:  # choose some randomized seed for parameter initialization
        args.seed = np.random.randint(1000000)
    return args
