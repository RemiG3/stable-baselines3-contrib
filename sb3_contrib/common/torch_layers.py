from typing import Dict, List, Tuple, Type, Union, Optional
from itertools import zip_longest
import torch
from torch import nn

from stable_baselines3.common.utils import get_device


class MlpNetwork(nn.Module):
    """
    Constructs a MLP that receives the output from a previous features extractor or directly the observations (if no features extractor is applied) as an input and outputs a latent representation for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[Dict[str, List[int]], List[Union[int, Dict[str, List[int]]]]],
        activation_fn: Union[List[Type[nn.Module]], Type[nn.Module]],
        layer_norm: Optional[Union[List[bool], bool]],
        dropout: Union[List[float], float],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        
        device = get_device(device)
        shared_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        policy_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim
        
        idx = 0
        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            policy_only_layers = net_arch["pi"]
            value_only_layers = net_arch["vf"]
        else:
            # Iterate through the shared layers and build the shared parts of the network
            for layer in net_arch:
                if isinstance(layer, int):  # Check that this is a shared layer
                    shared_net.append(nn.Linear(last_layer_dim_shared, layer))
                    if ((isinstance(layer_norm, list) and layer_norm[idx]) or ((not isinstance(layer_norm, list)) and layer_norm)):
                        shared_net.append( nn.LayerNorm(normalized_shape=layer) )
                    if ((isinstance(dropout, list) and (dropout[idx] > 0.)) or ((not isinstance(dropout, list)) and (dropout > 0.))):
                        shared_net.append( nn.Dropout(p=dropout[idx] if isinstance(dropout, list) else dropout) )
                    if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                        shared_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                    last_layer_dim_shared = layer
                    idx += 1
                else:
                    assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                    if "pi" in layer:
                        assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                        policy_only_layers = layer["pi"]

                    if "vf" in layer:
                        assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                        value_only_layers = layer["vf"]
                    break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for i, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                if ((isinstance(layer_norm, list) and layer_norm[idx]) or ((not isinstance(layer_norm, list)) and layer_norm)):
                    policy_net.append( nn.LayerNorm(normalized_shape=pi_layer_size) )
                if ((isinstance(dropout, list) and (dropout[idx] > 0.)) or ((not isinstance(dropout, list)) and (dropout > 0.))):
                    policy_net.append( nn.Dropout(p=dropout[idx] if isinstance(dropout, list) else dropout) )
                if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                    policy_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                if ((isinstance(layer_norm, list) and layer_norm[idx]) or ((not isinstance(layer_norm, list)) and layer_norm)):
                    value_net.append( nn.LayerNorm(normalized_shape=vf_layer_size) )
                if ((isinstance(dropout, list) and (dropout[idx] > 0.)) or ((not isinstance(dropout, list)) and (dropout > 0.))):
                    value_net.append( nn.Dropout(p=dropout[idx] if isinstance(dropout, list) else dropout) )
                if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                    value_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                last_layer_dim_vf = vf_layer_size
            
            idx += 1

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Union[List[Type[nn.Module]], Type[nn.Module]] = nn.ReLU,
    layer_norm: Union[List[bool], bool] = False,
    dropout: Union[List[float], float] = .0,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    layer_norm = [layer_norm] * len(net_arch) if isinstance(layer_norm, bool) else layer_norm
    dropout = [dropout] * len(net_arch) if isinstance(dropout, float) else dropout
    activation_fn = [activation_fn] * len(net_arch) if not isinstance(activation_fn, list) else activation_fn

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn[0](), nn.LayerNorm(normalized_shape=net_arch[0]) if layer_norm[0] else nn.Identity(), nn.Dropout(p=dropout[0])]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn[idx+1]())
        modules.append(nn.LayerNorm(normalized_shape=net_arch[idx + 1]) if layer_norm[idx+1] else nn.Identity())
        modules.append(nn.Dropout(p=dropout[idx+1]))

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# class MlpExtractor(nn.Module):
#     """
#     Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
#     the observations (if no features extractor is applied) as an input and outputs a latent representation
#     for the policy and a value network.

#     The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
#     It can be in either of the following forms:
#     1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
#         policy and value nets individually. If it is missing any of the keys (pi or vf),
#         zero layers will be considered for that key.
#     2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
#         in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
#         where int_list is the same for the actor and critic.

#     .. note::
#         If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

#     :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
#     :param net_arch: The specification of the policy and value networks.
#         See above for details on its formatting.
#     :param activation_fn: The activation function to use for the networks.
#     :param device: PyTorch device.
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         net_arch: Union[List[int], Dict[str, List[int]]],
#         activation_fn: Type[nn.Module],
#         dropout_actor: Union[float, List[float]] = 0., # If shared then dropout_actor is used for both actor and critic
#         dropout_critic: Union[float, List[float]] = 0.,
#         layer_norm_actor: Optional[bool] = False, # If shared then layer_norm_actor is used for both actor and critic
#         layer_norm_critic: Optional[bool] = False,
#         device: Union[th.device, str] = "auto",
#     ) -> None:
#         super().__init__()
#         device = get_device(device)
#         policy_net: List[nn.Module] = []
#         value_net: List[nn.Module] = []
#         last_layer_dim_pi = feature_dim
#         last_layer_dim_vf = feature_dim

#         # save dimensions of layers in policy and value nets
#         if isinstance(net_arch, dict):
#             # Note: if key is not specificed, assume linear network
#             pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
#             vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
#         else:
#             pi_layers_dims = vf_layers_dims = net_arch
#         # Iterate through the policy layers and build the policy net
#         for curr_layer_dim in pi_layers_dims:
#             policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
#             policy_net.append(activation_fn())
#             last_layer_dim_pi = curr_layer_dim
#         # Iterate through the value layers and build the value net
#         for curr_layer_dim in vf_layers_dims:
#             value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
#             value_net.append(activation_fn())
#             last_layer_dim_vf = curr_layer_dim

#         # Save dim, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf

#         # Create networks
#         # If the list of layers is empty, the network will just act as an Identity module
#         self.policy_net = nn.Sequential(*policy_net).to(device)
#         self.value_net = nn.Sequential(*value_net).to(device)

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.policy_net(features)

#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.value_net(features)

