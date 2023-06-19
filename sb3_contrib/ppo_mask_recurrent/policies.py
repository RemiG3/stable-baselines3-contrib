from sb3_contrib.common.maskable_recurrent.policies import (
    MaskableRecurrentActorCriticPolicy,
    MaskableRecurrentActorCriticCnnPolicy,
    MaskableRecurrentMultiInputActorCriticPolicy,
)

MlpPolicy = MaskableRecurrentActorCriticPolicy
CnnPolicy = MaskableRecurrentActorCriticCnnPolicy
MultiInputPolicy = MaskableRecurrentMultiInputActorCriticPolicy