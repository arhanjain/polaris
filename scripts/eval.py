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
import pandas as pd


from typing import List
from datetime import datetime
from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs

def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab 
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    import polaris.environments
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import MangerBasedRLSplatEnv
    from polaris.utils import load_eval_initial_conditions, run_folder_path
    from polaris.policy import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: MangerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)   # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    rollouts = len(initial_conditions)
    run_folder = run_folder_path(eval_args.run_folder, eval_args.environment, eval_args.policy.name)
    # Resume CSV logging
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame({
            'episode': pd.Series(dtype='int'),
            'episode_length': pd.Series(dtype='int'),
            'success': pd.Series(dtype='bool'),
            'progress': pd.Series(dtype='float'),
        })
    episode = len(episode_df)
    if episode >= rollouts:
        print(f"All rollouts have been evaluated. Exiting.")
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(object_positions = initial_conditions[episode % len(initial_conditions)])
    policy_client.reset()
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender)
        # obs, rew, term, trunc, info = env.step(torch.tensor(action).reshape(1, -1), expensive=True)

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                'episode': episode,
                'episode_length': bar.n,
                'success': info['rubric']['success'],
                'progress': info['rubric']['progress'],
            }
            episode_df = pd.concat([episode_df, pd.DataFrame([episode_data])], ignore_index=True)
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(object_positions = initial_conditions[episode % len(initial_conditions)])

            episode += 1
            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
