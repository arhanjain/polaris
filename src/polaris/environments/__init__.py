import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import MangerBasedRLSplatEnv
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import rubric system
from polaris.environments.rubrics import Rubric
from polaris.utils import DATA_PATH
import polaris.environments.rubrics.checkers as checkers


# =============================================================================
# Environment Registration
# =============================================================================

gym.register(
    id='DROID-RoboSplat',
    entry_point=MangerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
    },
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id='DROID-BlockStackKitchen',
    entry_point=MangerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "environments/block_stack_kitchen/g60_kitchen_table_zed.usd"),
        "rubric": Rubric (
            criteria=[
                checkers.reach("green_cube", threshold=0.2),
                checkers.reach("wood_cube", threshold=0.2),
                (checkers.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
                (checkers.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
                (checkers.is_within_xy("green_cube", "tray", 0.8), [2]),
                (checkers.is_within_xy("wood_cube", "tray", 0.8), [3]),
                (checkers.is_within_xy("green_cube", "wood_cube", 0.5), [4, 5]),
            ]
        ),
    },
    disable_env_checker=True,
    order_enforce=False,
)


# # =============================================================================
# # Rubric Registration
# # Define which rubric to use for each task (identified by USD stem name)
# # =============================================================================

# # Block stacking tasks
# register_rubric(
#     "g60_kitchen_table_zed",  # block_stack_kitchen
#     StackingRubric,
#     stack_order=["blue_block", "red_block"],  # blue on bottom, red on top
#     xy_tolerance=0.04,
# )

# register_rubric(
#     "vention",  # block_stack
#     StackingRubric,
#     stack_order=["green_block", "blue_block"],
#     xy_tolerance=0.04,
# )

# # Food bussing - move food items to target zone (e.g., plate/bin)
# register_rubric(
#     "g60_corner",  # food_bussing
#     ObjectInZoneRubric,
#     targets=[
#         {"object": "banana", "zone_center": (0.3, 0.0, 0.1), "zone_radius": 0.1},
#         {"object": "apple", "zone_center": (0.3, 0.0, 0.1), "zone_radius": 0.1},
#     ],
# )

# # Pan cleaning - move pan to target location
# register_rubric(
#     "g60_stovetop_zed",  # pan_clean
#     ObjectInZoneRubric,
#     targets=[
#         {"object": "stainlesspan", "zone_center": (0.4, -0.2, 0.05), "zone_radius": 0.15},
#     ],
# )

# # Bowl stacking
# register_rubric(
#     "envs_00",  # stack_bowls
#     StackingRubric,
#     stack_order=["pink_bowl", "yellow_bowl", "blue_bowl"],
#     xy_tolerance=0.06,
# )

# # Add more rubrics as needed...
# # register_rubric("task_usd_stem", RubricClass, **config)
