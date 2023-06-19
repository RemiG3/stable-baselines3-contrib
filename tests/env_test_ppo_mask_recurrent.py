from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable_recurrent.evaluation import evaluate_policy
from sb3_contrib.common.maskable_recurrent.policies import MaskableRecurrentActorCriticPolicy
from sb3_contrib import MaskableRecurrentPPO

env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
policy_kwargs = dict(lstm_hidden_size=256,
                     n_lstm_layers=1,
                     shared_lstm=False,
                     enable_critic_lstm=True,
                     lstm_kwargs=dict(dropout=0.1))
model = MaskableRecurrentPPO(MaskableRecurrentActorCriticPolicy, env, gamma=0.4, seed=32, verbose=1)#, policy_kwargs=policy_kwargs)
model.learn(100)

evaluate_policy(model, env, n_eval_episodes=20, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskableRecurrentPPO.load("ppo_mask")

obs = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, rewards, dones, info = env.step(action)
    env.render()