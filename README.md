# PolaRiS

PolaRiS is a evaluation framework for generalist policies. It provides tooling for reconstructing environments, evaluating models, and running experiments with minimal setup.

## Installation

### Clone the repository (recursively)

```bash
git clone --recursive git@github.com:arhanjain/PolaRiS.git
cd PolaRiS
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Setup environment with uv
If you don't have UV installed, see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
```bash
uv sync
```

<!-- ### HuggingFace Datasets
For using our evaluation DROID environments or simulation cotraining data, clone the datasets below.
```bash
# Environments
uvx hf download owhan/PolaRiS-environments --repo-type=dataset --local-dir ./PolaRiS-environments
# Cotrain Datasets
uvx hf download owhan/PolaRis-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets
``` -->

## Getting Started
### Minimal Code Example
Initializing a PolaRiS environment and executing random actions
```python
import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
# This must be done before importing anything with dependency on Isaaclab
# >>>> Isaac Sim App Launcher <<<<
parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# >>>> Isaac Sim App Launcher <<<<

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import MangerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

env_cfg = parse_env_cfg(
    "DROID-FoodBussing",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)
env: MangerBasedRLSplatEnv = gym.make("DROID-FoodBussing", cfg=env_cfg)   # type: ignore
language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions = initial_conditions[0])

while True:
    action = torch.tensor(env.action_space.sample())
    obs, rew, term, trunc, info = env.step(action, expensive=True)

    if term[0] or trunc[0]:
        break

print(f"Episode Finished. Success: {info['rubric']['success']}, Progress: {info['rubric']['progress']}")
```

### Run a π0.5 Policy in PolaRiS
*Note: First run may take longer due to JIT compilation of the splat rasterization kernels*
```bash
# Install evaluation environments (<2GB) and start evaluation process
uvx hf download owhan/PolaRiS-environments --repo-type=dataset --local-dir ./PolaRiS-environments
uv run scripts/eval.py --environment DROID-FoodBussing --policy.name pi05 --policy.client DroidJointPos --policy.port 8000 --policy.open-loop-horizon 8

# In separate process, starting from the root of this repo. This will setup openpi and host a pi05 policy.
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_droid_jointpos_fullfinetune --policy.dir gs://openpi-assets-simeval/pi05_droid_jointpos
```
By default, if no run folder is specified, results will be stored in a run folder created under `runs/{YYYY-MM-DD}/{HH:MM:SS AM/PM}`

For the full list of all checkpoints we provide for evaluation, see [checkpoints.md](docs/checkpoints.md)

**TODO**: replace the policy path here to hosted sim finetuned checkpoints


### Batch Evaluation
Running a full scale evaluation across multiple checkpoints and tasks can be easily configured with a single python file representing the entire experiment. You can optionally name your experiments via `--run-folder` flag. For example configs, see [experiments/example.py](experiments/example.py)
```bash
uv run scripts/batch_eval.py --config experiments/example.py --run-folder runs/i-love-robots
```

## Creating Custom Evaluation Environments (Time Estimate: XX)
**TODO**

## Adding Policies to Evaluate
**TODO**

## Project Structure

<!-- ```text
PolaRiS/
├── scripts/
│   └── eval.py
├── PolaRiS-environments/
├── PolaRiS-datasets/
├── src/polaris/
└── README.md
``` -->


TODO
- If nvcc, cuda toolkit isnt installed, what to do
- supports CUDA 12 only
- make sure the TORCH archirecutre list is correct (mineby default included way more than it needed)
- have correct version of gxx (my versions was too new)
- clear torch_extensions cache in between builds and env changes