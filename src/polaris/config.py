"""
Lightweight config dataclasses for evaluation.
No heavy dependencies - safe to import anywhere.
"""

from dataclasses import dataclass


@dataclass
class PolicyServer:
    """
    Configuration for a policy server to co-launch.

    Use {port} placeholder in command - it will be replaced with an auto-assigned free port.
    Jobs using this server will automatically have their policy.port updated.

    Example:
        PolicyServer(
            name="pi0",
            command="CUDA_VISIBLE_DEVICES=0 python serve_policy.py --port {port}",
        )
    """

    name: str  # Friendly name for logging (also used to match jobs to servers)
    command: str  # Shell command with {port} placeholder
    ready_message: str = (
        "Application startup complete"  # Message indicating server is ready
    )

    # Runtime-assigned (don't set manually)
    _assigned_port: int | None = None


@dataclass
class PolicyArgs:
    """Policy configuration."""

    # name: str                              # Policy name (pi05_droid_jointpos, pi0_fast_droid_jointpos, etc.)
    client: str = "DroidJointPos"  # Client name (DroidJointPos, Fake, etc.)
    host: str = "0.0.0.0"
    port: int = 8000
    open_loop_horizon: int | None = 8


@dataclass
class EvalArgs:
    """Evaluation configuration."""

    policy: PolicyArgs  # Policy arguments
    environment: str  # Which IsaacLab environment to use
    run_folder: str  # Path to run folder
    headless: bool = True  # Whether to run in headless mode
    initial_conditions_file: str | None = None  # Path to initial conditions file
    instruction: str | None = None  # Override language instruction
    rollouts: int | None = None  # Number of rollouts to evaluate


@dataclass
class JobCfg:
    """A single evaluation job in a batch."""

    eval_args: EvalArgs
    server: PolicyServer | None = None  # Server to co-launch for this job


@dataclass
class BatchConfig:
    """Batch evaluation configuration."""

    jobs: list[JobCfg]

    # @staticmethod # let users do this on their own if they want
    # def sweep(**kwargs: list[Any]) -> list[dict[str, Any]]:
    #     """
    #     Helper to generate grid of configs from lists of values.

    #     Example:
    #         BatchConfig.sweep(
    #             usd=["env1.usd", "env2.usd"],
    #             policy=["pi0", "pi05"],
    #         )
    #         # Returns 4 dicts: all combinations
    #     """
    #     keys = list(kwargs.keys())
    #     values = [v if isinstance(v, list) else [v] for v in kwargs.values()]
    #     return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
