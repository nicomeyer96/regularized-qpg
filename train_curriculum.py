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
import os
import pickle
import warnings
import torch
import time
import numpy as np
from torch.distributions import Categorical
from functools import partial
from qiskit_torch_module import QuantumModule

from utils import parse, CartPoleWrapper

from qrl_helpers.buffer import Buffer
from qrl_helpers.circuit import generate_circuit
from qrl_helpers.employ import employ_model
from qrl_helpers.loss import regularized_policy_gradient


VELOCITY, VELOCITY_PM = iter([0.0, 0.5, 1.0, 1.5, None]), 0.25


def curriculum_regularized_quantum_policy_gradients(env_name: str, discount_rate: float, vqc_layers: int, seed: int,
                                                    episodes: int, batch_size: int, lipschitz_regularization: float,
                                                    val_freq: int, threshold: float, threads: int, max_failures: int):
    """ Realizes the quantum policy gradient algorithm described in
        N. Meyer et al., Quantum Policy Gradient Algorithm with Optimized Action Decoding, PMLR 202:24592-24613, 2023.
        with additional Lipschitz regularization in a curriculum-training manner using the qiskit-torch-module.

        Args:
            env_name: Which environment to train on, either `CartPole-v0` or `CartPole-v1`
            discount_rate: MDP discount rate
            vqc_layers: Number of ansatz layers
            seed: Random seed for parameter initialization
            episodes: Number of training episodes
            batch_size: Number of environments trained on in each epoch
            lipschitz_regularization: Weighting for Lipschitz regularization
            val_freq: Frequency of validation to determine early stopping criterion
            threshold: Early stopping threshold
            threads: Number of parallel training threads
            max_failures: Maximum number of allowed failures
    """

    # extract initial velocity range
    velocity_range = next(VELOCITY)
    # initialize environments for training (batch_size) and validation (100)
    env = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode='sync',
                       wrappers=[partial(CartPoleWrapper, pole_velocity=(-(velocity_range+VELOCITY_PM),
                                                                         velocity_range+VELOCITY_PM))])
    env_val = gym.make_vec(env_name, num_envs=100, vectorization_mode='sync',
                           wrappers=[partial(CartPoleWrapper, pole_velocity=(-(velocity_range+VELOCITY_PM),
                                                                             velocity_range+VELOCITY_PM))])

    # data buffers for trajectories
    buffer = Buffer(batch_size, gamma=discount_rate)
    buffer_val = Buffer(100)

    # generate VQC used for policy approximation
    vqc, params_encoding, (params_variational, params_scaling) = generate_circuit(num_qubits=env.observation_space.shape[1],
                                                                                  depth=vqc_layers,
                                                                                  entanglement_structure='full',
                                                                                  input_scaling=True)

    # quantum neural network that handles gradient computation via PyTorch
    # initialize variational parameters with Normal(mean=0.0, std=0.1); initialize scaling parameters to 1.0
    model = QuantumModule(vqc, encoding_params=params_encoding, variational_params=[params_variational, params_scaling],
                          variational_params_names=['variational', 'scaling'],
                          variational_params_initial=[('normal', {'std': 0.1}), ('constant', {'val': 1.0})],
                          observables='tensoredZ', num_threads_forward=threads, num_threads_backward=threads,
                          seed_init=seed)

    opt = torch.optim.Adam([{'params': model.variational, 'lr': 0.05}, {'params': model.scaling, 'lr': 0.05}],
                           amsgrad=True)

    epochs = episodes // batch_size
    print('Curriculum training for {} Epochs with a batch size of {} on initial angular velocity range [{}, {}].\n'
          .format(epochs, batch_size, -(velocity_range+VELOCITY_PM), (velocity_range+VELOCITY_PM)))

    start_time = time.perf_counter()
    time_validation = 0.0

    failure_counter = 0

    # start training
    for epoch in range(epochs):

        if failure_counter >= max_failures:
            print(f'Maximum failure threshold reached, terminating after {epoch} epochs.')
            break

        # handle validation and early stopping
        if 0 == epoch % val_freq:
            start_time_validation = time.perf_counter()
            val_reward = employ_model(env_val, buffer_val, model)
            time_validation += time.perf_counter() - start_time_validation
            print('VALIDATE [Epoch {}] Avg. Reward: {:.2f}     (max: {:.0f}, min: {:.0f})'
                  .format(epoch, *val_reward))
            # test if average reward is over early stopping threshold
            if val_reward[0] >= threshold:  # noqa
                print(f'\nThe desired reward value was achieved for angular velocity range '
                      f'[{-(velocity_range+VELOCITY_PM):.2f},{(velocity_range+VELOCITY_PM):.2f}].')
                path = (f'curr[{-(velocity_range+VELOCITY_PM):.2f},{(velocity_range+VELOCITY_PM):.2f}]'
                        f'_reg={lipschitz_regularization:.2f}_{seed}.pkl')
                result = {
                    'args': _args,
                    'rewards_train': buffer.get_running_rewards,
                    'rewards_val': buffer_val.get_running_rewards,
                    'total_steps': buffer.get_total_steps,
                    'parameters': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'velocity_range': velocity_range,
                    'failures': failure_counter
                }
                path = os.path.join('results', 'train', path)
                with open(path, 'wb') as ff:
                    pickle.dump(result, ff)
                print(f'Saved intermediate result to `{path}`.')
                # increase velocity range
                velocity_range = next(VELOCITY)
                if velocity_range is None:
                    print(f'Maximum angular velocity range reached, terminating after {epoch} epochs.')
                    break
                # update environment reset strategy
                env = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode='sync',
                                   wrappers=[partial(CartPoleWrapper, pole_velocity=(-(velocity_range + VELOCITY_PM),
                                                                                     velocity_range + VELOCITY_PM))])
                env_val = gym.make_vec(env_name, num_envs=100, vectorization_mode='sync',
                                       wrappers=[
                                           partial(CartPoleWrapper, pole_velocity=(-(velocity_range + VELOCITY_PM),
                                                                                   velocity_range + VELOCITY_PM))])
                print(f'Increased angular velocity range to [{-(velocity_range+VELOCITY_PM):.2f}, '
                      f'{velocity_range+VELOCITY_PM}:.2f].')

        buffer.reset()

        # it is possible to use random seeds here in order to get completely reproducible behavior
        observations, _ = env.reset()

        # compute forward passes for data generation sequentially, as batch_size typically to low as that
        # parallelization brings an advantage
        num_threads_forward = model.num_threads_forward
        model.set_num_threads_forward(1)

        while True:  # while not all environments have terminated

            # store observations
            buffer.append_observations(observations)

            # we do not record gradients as of now, this allows for a much faster batch-parallel backward pass later on
            with torch.set_grad_enabled(False):
                # sample action following current policy -- for entire batch at once
                # policy: [pi(a=0|obs), pi(a=1|obs)] = [(model(obs) + 1) / 2, (-model(obs) + 1) / 2]
                policies = torch.div(torch.add(torch.mul(model(observations).repeat(1, 2), torch.tensor([1.0, -1.0])), 1.0), 2.0)
                # sample action from distribution defined by policy
                dist = Categorical(policies)
                actions = [action.item() for action in dist.sample()]

            # store actions, returns 0-padded actions (as some environments might have already terminated)
            # (this is a small workaround, as gym.VectorEnv automatically resets terminated environments)
            actions_padded = buffer.append_actions(actions)

            # execute actions in respective environments
            observations_, rewards, terminated, truncated, _ = env.step(actions_padded)

            # store rewards
            buffer.append_rewards(rewards)

            # update termination and truncation flags
            # returns indices of environments that are still not done -> use to select only relevant observations
            indices_not_done = buffer.update_terminated_and_truncated(terminated, truncated)
            observations = observations_[indices_not_done]

            # check if all environments are done
            if 0 == len(indices_not_done):
                buffer.finish_epoch()
                failure_counter += sum(200.0 > np.array(buffer.epoch_rewards))
                break

        # set back to parallel computation for computing gradients
        model.set_num_threads_forward(num_threads_forward)

        # print some statistics
        print('CURRICULUM TRAIN [Epoch {}->{}, Episodes {}-{}, Total Steps: {}, Failures: {}]  Avg. Reward: {:.1f}'
              '    (max: {:.0f}, min: {:.0f})'.format(epoch, epoch+1, epoch*batch_size+1, (epoch+1) * batch_size,
                                                      buffer.get_total_steps, failure_counter, *buffer.get_epoch_reward))

        # batched forward pass
        batched_policies = torch.div(torch.add(torch.mul(model(buffer.batched_observations).repeat(1, 2),
                                                         torch.tensor([1.0, -1.0])), 1.0), 2.0)
        # filter by executed action, compute log
        batched_log_policies = torch.log(torch.gather(batched_policies, 1,
                                                      torch.LongTensor(buffer.batched_actions).unsqueeze(1)))

        # optimize parameters with acquired data
        opt.zero_grad()
        loss = regularized_policy_gradient(batched_log_policies, buffer.batched_rewards, model.scaling,
                                           lamb=lipschitz_regularization)
        loss.backward()
        opt.step()

    time_total = time.perf_counter() - start_time

    result = {
        'args': _args,
        'rewards_train': buffer.get_running_rewards,
        'rewards_val': buffer_val.get_running_rewards,
        'total_steps': buffer.get_total_steps,
        'parameters': model.state_dict(),
        'optimizer': opt.state_dict(),
        'time_train': time_total - time_validation,
        'time_val': time_validation,
        'velocity_range': velocity_range,
        'failures': failure_counter
    }

    path = os.path.join('results', 'train', f'curr_reg={lipschitz_regularization:.2f}_{seed}.pkl')
    with open(path, 'wb') as ff:
        pickle.dump(result, ff)

    print('\nTotal time: {:.1f}s     (thereof {:.1f}s for validation)'.format(time_total, time_validation))


if __name__ == '__main__':
    _args = parse()
    if 0 >= _args.val:
        raise ValueError('Running in curriculum training mode requires setting `--val` to a positive integer value.')
    if 1 != _args.val:
        warnings.warn('We recommend to use `--val=1` for curriculum QRL.')
    print(f'Storing results to results/train/curr_reg={_args.reg:.2f}_{_args.seed}.pkl')
    curriculum_regularized_quantum_policy_gradients(_args.env,
                                                    discount_rate=_args.gamma,
                                                    vqc_layers=_args.layers,
                                                    seed=_args.seed,
                                                    episodes=_args.episodes,
                                                    batch_size=_args.batch,
                                                    lipschitz_regularization=_args.reg,
                                                    val_freq=_args.val,
                                                    threshold=195.0 if 'CartPole-v0' == _args.env else 475.0,
                                                    threads=_args.threads,
                                                    max_failures=_args.failures)
    print(f'Results stored to results/train/curr_reg={_args.reg:.2f}_{_args.seed}.pkl')
