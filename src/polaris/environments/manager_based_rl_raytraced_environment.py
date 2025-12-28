import torch
import cv2
from pathlib import Path
import numpy as np

from isaaclab.sensors.camera.camera import Camera
import isaaclab.utils.math as math
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.stage import get_current_stage
from pxr import Semantics

from polaris.environments.rubrics import Rubric


class ManagerBasedRLRaytracedEnv(ManagerBasedRLEnv):
    rubric: Rubric | None = None
    _task_name: str | None = None

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        *args,
        rubric: Rubric | None = None,
        usd_file: str | None = None,
        **kwargs,
    ):
        # do dynamic setup here maybe
        if usd_file is not None:
            self.usd_file = usd_file
            cfg.dynamic_setup(usd_file)

        super().__init__(cfg=cfg, *args, **kwargs)
        self.rubric = rubric

    def _evaluate_rubric(self) -> dict:
        """Evaluate rubric and return results for info dict."""
        if self.rubric is None:
            return {
                "rubric": {
                    "success": False,
                    "progress": -1.0,
                    "metrics": {},
                }
            }

        result = self.rubric.evaluate(self)
        return {
            "rubric": {
                "success": result.success,
                "progress": result.progress,
                "metrics": result.metrics,
            }
        }

    def reset(self, object_positions: dict = {}, expensive=True, *args, **kwargs):
        """
        Reset the environment

        Parameters
        ----------
        object_positions : dict
            A dictionary mapping object names to their desired poses (position and orientation).
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, info = super().reset(*args, **kwargs)

        # Reset rubric state
        if self.rubric:
            self.rubric.reset()

        # Following predefined initial conditions
        for obj, pose in object_positions.items():
            print(f"Setting initial condition for {obj} to {pose}")
            pose = torch.tensor(pose)[None]
            self.scene[obj].write_root_pose_to_sim(pose)
        self.sim.render()
        self.scene.update(0)
        obs = (
            self.observation_manager.compute()
        )  # update observation after setting ICs if needed

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, info

    def step(self, action, expensive=True):
        """
        Steps the environment

        Parameters
        ----------
        action: torch.Tensor
            The action to take in the environment.
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, rew, done, trunc, info = super().step(action)

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, rew, done, trunc, info
