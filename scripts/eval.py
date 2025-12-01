import sys
import tyro
import mediapy
# import wandb
import tqdm
import gymnasium as gym
import cv2
import torch
import argparse
import numpy as np
import random
import time
import json


from typing import List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from isaaclab.app import AppLauncher
import polaris.policy as policy_

@dataclass
class EvalArgs:
    usd: str
    policy: policy_.PolicyArgs 
    headless: bool = True
    environment: str = "DROID-RoboSplat"
    # instructions: List[str] = field(default_factory=lambda: []) # TODO 
    # rollouts: int = 10 # TODO: make this the number of initial conditions provided

    run_folder: str | None = None
    # visualize: bool = False
    # object_positions: List[dict] = field(default_factory=lambda: [{}]) # [{object_name: (x, y, z, qw, qx, qy, qz)}]
    # object_positions_file: str | None = None
    # platform: str = "DROID"
    # robot_splat: bool = False
    # diffusion: bool = False
    # pure_sim: bool = False
    # nightmare: bool = False

def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab 
    # Inside main function for compatibility with HPC cluster python launch scripts
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    args_cli, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args  # clear out sys.argv for hydra
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    import polaris.environments
    from polaris.utils import parse_env_cfg
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        usd_file=eval_args.usd,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    env = gym.make(eval_args.environment, cfg=env_cfg)


    run_folder = eval_args.run_folder
    if not run_folder:
        run_folder = f"runs/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%I:%M:%S %p')}"

    run_folder = (
        Path(run_folder)
        / Path(eval_args.usd).stem
        / eval_args.policy
    )
    print(f" >>> Saving to {run_folder} <<< ")
    run_folder.mkdir(parents=True, exist_ok=True)
    episode = len(
        list(run_folder.glob("*.mp4"))
    )  # check if rollouts exist in run_folder
    # if episode >= eval_args.rollouts:
    if episode >= 5:
        print(f"All rollouts have been evaluated. Exiting.")
        return
    print(f" >>> Starting eval job from episode {episode} <<< ")

    video = []
    horizon = env.unwrapped.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    successes = 0.0
    # obs, info = env.reset(object_positions = object_positions[episode % len(object_positions)], expensive=not eval_args.pure_sim)
    obs, info = env.reset()

    # curr_instruction = random.choice(eval_args.instructions)
    # client.reset(task_description=curr_instruction)
    # success_labeler.reset(curr_instruction)
    while True:
        obs, rew, term, trunc, info = env.step(torch.zeros(1, env.unwrapped.action_space.shape[1]), expensive=True)

        external_cam = obs["splat"]["external_cam"]

        from PIL import Image

        im = Image.fromarray(external_cam)
        im.save("test.png")
        break

    env.close()
    simulation_app.close()

        # cv2.imshow("test", external_cam)
        # cv2.waitKey(1)




if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
