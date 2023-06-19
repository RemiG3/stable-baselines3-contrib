import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import torch
from torch import nn

env = gym.make('ChargingStationEnv-v0',
               schema="template.json",
               initializer=getattr(getattr(getattr(__import__('Charging_Station_Env'), 'utils'), 'initializer'), "Initializer")(datafile="Generated sample (AC,poisson_fit) Horizon =2015-01-01-to-2015-01-02.csv", energy_initializer=getattr(getattr(getattr(__import__('Charging_Station_Env'), 'utils'), 'initializer'), "Energy_Initializer")()),
               simulation_controller=getattr(getattr(getattr(__import__('Charging_Station_Env'), 'utils'), 'station_simulator'), "Simulate_Station")(),
               action_controller=getattr(getattr(getattr(__import__('Charging_Station_Env'), 'utils'), 'actions_simulator'), "Simulate_Actions")(),
              )


#check_env(env)

policy_kwargs = dict(activation_fn=nn.Tanh,
                     net_arch=[64, dict(pi=[32], vf=[32])],
                     layer_norm=True,
                     dropout=True,
					 n_lstm_layers=3,
					 lstm_hidden_size=128,
					 shared_lstm=False,
					 enable_critic_lstm=True,
					 lstm_kwargs=dict(dropout=.1, bias=True, bidirectional=False),
					 dropout_actor=0.,
					 dropout_critic=0.,
					 layer_norm_actor=True,
					 layer_norm_critic=True)

model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log=None, policy_kwargs=policy_kwargs, learning_rate=0.001, n_steps=1000, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0., vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, seed=0, device='auto', _init_setup_model=True)


"""
policy_kwargs = dict(activation_fn=nn.Tanh,
                     net_arch=[64, dict(pi=[32], vf=[32])])

model = PPO('MlpPolicy', env, verbose=1)#, tensorboard_log=None, policy_kwargs=policy_kwargs, learning_rate=0.001, n_steps=1000, batch_size=64, n_epochs=10, gamma=0.99)#, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0., vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, seed=0, device='auto', _init_setup_model=True)
"""

print(model.policy)

model.learn(total_timesteps=10, reset_num_timesteps=False, tb_log_name='logs/test')

env.close()