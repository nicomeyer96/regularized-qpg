# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import os
import pickle
import time
import numpy as np
import gymnasium as gym
from functools import partial
from qiskit_torch_module import QuantumModule

from utils import parse, CartPoleWrapper

from qrl_helpers.buffer import Buffer
from qrl_helpers.circuit import generate_circuit
from qrl_helpers.employ import employ_model

# perturbation values to test for
PERTURBATIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# angle values and offset to test for
ANGLES, ANGLES_PM = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25], 0.01
# velocity values and offset to test for
VELOCITIES, VELOCITIES_PM = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 0.25


def run_test(env_name: str, model: QuantumModule, runs: int = 100, perturbation: float = 0.0,
             angle: float = None, velocity: float = None):
    """ Test pretrained models for different perturbation and initial angle / velocity values

        Args:
            env_name: Which environment to train on, either `CartPole-v0` or `CartPole-v1`
            model: Trained model
            runs: Number of runs to test for
            perturbation: Perturbation of environment
            angle: Angle to reset environment to
            velocity: Angular velocity to reset environment to
    """

    # initialize environment with perturbation or specific angle / velocity initial ranges
    env = gym.make_vec(env_name, num_envs=runs, vectorization_mode='sync',
                       wrappers=[partial(CartPoleWrapper, perturbation=perturbation,
                                         pole_angle=None if angle is None else (angle-ANGLES_PM,
                                                                                angle+ANGLES_PM),
                                         pole_velocity=None if velocity is None else (velocity-VELOCITIES_PM,
                                                                                      velocity+VELOCITIES_PM))])

    buffer = Buffer(runs)
    buffer.reset()

    employ_model(env, buffer, model)
    return np.array(buffer.epoch_rewards)


def test(model_path, runs, threads=0, perturbate=True, angle=False, velocity=False):
    test_results = {}

    # load trained model parameters
    if model_path is None:
        raise ValueError('Trained model has to be specified.')
    model_path_full = os.path.join('results', 'train', f'{model_path}.pkl')
    if not os.path.exists(model_path_full):
        raise RuntimeError(f'Trained model `{model_path_full}` does not exist.')
    model_data = pickle.load(open(model_path_full, 'rb'))

    # set up model
    env = CartPoleWrapper(gym.make(model_data['args'].env))
    vqc, params_encoding, (params_variational, params_scaling) = generate_circuit(
        num_qubits=env.observation_space.shape[0],
        depth=model_data['args'].layers,
        entanglement_structure='full',
        input_scaling=True)
    model = QuantumModule(vqc, encoding_params=params_encoding,
                          variational_params=[params_variational, params_scaling],
                          variational_params_names=['variational', 'scaling'],
                          observables='tensoredZ', num_threads_forward=threads)
    model.load_state_dict(model_data['parameters'])

    start_time = time.perf_counter()

    # select respective testing modes
    if perturbate:
        assert not angle and not velocity, 'Perturbation and angle / velocity initialization are exclusive'
        for perturbation in PERTURBATIONS:
            epoch_rewards = run_test(model_data['args'].env, model, runs=runs, perturbation=perturbation)
            print(f'[Perturbation={perturbation:.2f}]  Avg: {np.average(epoch_rewards):.2f},  '
                  f'Stable: {sum(200 == epoch_rewards)}/{runs}')  # noqa
            test_results[perturbation] = epoch_rewards
        path = os.path.join('robustness', f'{model_path}.pkl')
    elif angle is True and velocity is False:
        for _angle in ANGLES:
            epoch_rewards = run_test(model_data['args'].env, model, runs=runs, angle=_angle)
            print(f'[Angle={_angle:.2f}+-{ANGLES_PM:.2f}]  Avg: {np.average(epoch_rewards):.2f},  '
                  f'Stable: {sum(200 == epoch_rewards)}/{runs}')  # noqa
            test_results[_angle] = epoch_rewards
        path = os.path.join('generalization', f'angle_{model_path}.pkl')
    elif angle is False and velocity is True:
        for _velocity in VELOCITIES:
            epoch_rewards = run_test(model_data['args'].env, model, runs=runs, velocity=_velocity)
            print(f'[Velocity={_velocity:.2f}+-{VELOCITIES_PM:.2f}]  '
                  f'Avg: {np.average(epoch_rewards):.2f},  Stable: {sum(200 == epoch_rewards)}/{runs}')  # noqa
            test_results[_velocity] = epoch_rewards
        path = os.path.join('generalization', f'velocity_{model_path}.pkl')
    elif angle is True and velocity is True:
        for _angle in ANGLES:
            test_results_tmp = {}
            for _velocity in VELOCITIES:
                epoch_rewards = run_test(model_data['args'].env, model, runs=runs, angle=_angle, velocity=_velocity)
                print(f'[Angle={_angle:.2f}+-{ANGLES_PM:.2f}, '
                      f'Velocity={_velocity:.2f}+-{VELOCITIES_PM:.2f}]  '
                      f'Avg: {np.average(epoch_rewards):.2f},  Stable: {sum(200 == epoch_rewards)}/{runs}')  # noqa
                test_results_tmp[_velocity] = epoch_rewards
            test_results[_angle] = test_results_tmp
        path = os.path.join('generalization', f'{model_path}.pkl')
    else:
        raise ValueError('Either `--perturbate`, `--angle`, or `--velocity` has to be set.')

    test_results['time'] = time.perf_counter() - start_time
    with open(os.path.join('results', path), 'wb') as ff:
        pickle.dump(test_results, ff)

    print(f'Saved testing results to `{os.path.join('results', path)}`.')


if __name__ == '__main__':
    _args = parse()
    test(_args.model, _args.runs,
         threads=_args.threads, perturbate=_args.perturbate, angle=_args.angle, velocity=_args.velocity)
