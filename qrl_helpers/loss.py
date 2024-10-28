# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import torch


def regularized_policy_gradient(log_policies, rewards, weights_scaling, lamb: float = 0.0):
    """ Compute regularized policy gradient. We need to use the negative value, as torch.optim by default does gradient
    descend (but policy gradients algorithms require gradient ascend).
    """
    # compute `standard` loss
    loss = torch.mean(torch.mul(log_policies.squeeze(1), torch.tensor(rewards)))
    # compute lipschitz term, factor of 1/2 to be consistent with paper definition
    reg_lipschitz = torch.sum(torch.square(torch.mul(weights_scaling, 0.5)))
    return torch.neg(torch.sub(loss, torch.mul(reg_lipschitz, lamb)))
