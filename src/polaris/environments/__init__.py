import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import MangerBasedRLSplatEnv
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv
# from .droid_uw_cfg import DroidUWCfg
# from .bridge_cfg import EnvCfg as BridgeCfg
# from .droid_cfg import EnvCfg as DroidCfg, EnvRelCfg as DroidRelCfg


gym.register(
        id='DROID-RoboSplat',
        entry_point=MangerBasedRLSplatEnv,
        kwargs={
            "env_cfg_entry_point": DroidCfg,
        },
        disable_env_checker=True,
        order_enforce = False,
        )
# gym.register(
#         id='DROID-nosplat',
#         entry_point=MangerBasedRLSplatEnv,
#         kwargs={
#             "env_cfg_entry_point": DroidCfg,
#         },
#         disable_env_checker=True
#         )
# gym.register(
#         id='DROID-Rel',
#         entry_point=BaseEvalEnv2,
#         kwargs={
#             "env_cfg_entry_point": DroidRelCfg,
#         },
#         disable_env_checker=True
#         )
#
# gym.register(
#     id="DroidUW",
#     entry_point=BaseEvalEnv,
#     kwargs={
#         "env_cfg_entry_point": DroidUWCfg,
#     },
#     disable_env_checker=True,
# )
