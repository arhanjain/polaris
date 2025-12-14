import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import (
    MangerBasedRLSplatEnv,
)
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
    id="DROID-RoboSplat",
    entry_point=MangerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
    },
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DROID-BlockStackKitchen",
    entry_point=MangerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "block_stack_kitchen/scene.usda"),
        "rubric": Rubric(
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


gym.register(
    id="DROID-FoodBussing",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "food_bussing/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("ice_cream_", threshold=0.2),
                checkers.reach("grapes", threshold=0.2),
                (checkers.lift("ice_cream_", threshold=0.06), [0]),
                (checkers.lift("grapes", threshold=0.06), [1]),
                (
                    checkers.is_within_xy("ice_cream_", "bowl", percent_threshold=0.8),
                    [2],
                ),
                (checkers.is_within_xy("grapes", "bowl", percent_threshold=0.8), [3]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PanClean",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pan_clean/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("sponge", threshold=0.2),
                (checkers.lift("sponge", threshold=0.09, default_height=0.0), [0]),
                (checkers.is_within_xy("sponge", "pan", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


gym.register(
    id="DROID-MoveLatteCup",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "move_latte_cup/scene.usda"),
    },
)

gym.register(
    id="DROID-OrganizeTools",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "organize_tools/scene.usda"),
    },
)

gym.register(
    id="DROID-TapeIntoContainer",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "tape_into_container/scene.usda"),
    },
)

gym.register(
    id="DROID-Test",
    entry_point=MangerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "test/scene.usda"),
    },
)
