"""
Example experiment config that co-launches policy servers with each job.

Usage:
    python scripts/batch_eval.py --config experiments/with_servers.py
    python scripts/batch_eval.py --config experiments/with_servers.py --dry-run
"""

from polaris.config import EvalArgs, PolicyArgs, BatchConfig, PolicyServer, JobCfg


# Define reusable servers - {port} is auto-replaced with a free port
PI0_SERVER = PolicyServer(
    name="pi0_fast",
    command=" ".join([
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35",
        "~/projects/PolaRiS/third_party/openpi/.venv/bin/python",
        "~/projects/PolaRiS/third_party/openpi/scripts/serve_policy.py",
        "--port {port}",
        "policy:checkpoint --policy.config pi0_fast_droid_jointpos",
        "--policy.dir gs://openpi-assets-simeval/pi0_fast_droid_jointpos",
    ]),
    ready_message="server listening on",
)

PI05_SERVER = PolicyServer(
    name="pi05",
    command=" ".join([
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35",
        "~/projects/PolaRiS/third_party/openpi/.venv/bin/python",
        "~/projects/PolaRiS/third_party/openpi/scripts/serve_policy.py",
        "--port {port}",
        "policy:checkpoint --policy.config pi05_droid_jointpos",
        "--policy.dir gs://openpi-assets-simeval/pi05_droid_jointpos",
    ]),
    ready_message="server listening on",
)

# Each job references its server
config = BatchConfig(
    jobs=[
        JobCfg(
            server=PI0_SERVER,
            eval_args=EvalArgs(
                environment="DROID-BlockStackKitchen",
                policy=PolicyArgs(
                    name="pi0_fast_droid_jointpos",
                    client="DroidJointPos",
                    open_loop_horizon=8,
                ),
                initial_conditions_file="PolaRiS-assets/environments/block_stack_kitchen/initial_conditions_silly.json"
            ),
        ),

        # # pi0 jobs
        # JobCfg(
        #     server=PI0_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/block_stack_kitchen/g60_kitchen_table_zed.usd",
        #         policy=PolicyArgs(
        #             name="pi0_fast_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/block_stack_kitchen/initial_conditions_silly.json"
        #     ),
        # ),
        # JobCfg(
        #     server=PI0_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/food_bussing/g60_corner.usd",
        #         policy=PolicyArgs(
        #             name="pi0_fast_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/block_stack_kitchen/initial_conditions_silly.json"
        #     ),
        # ),
        # JobCfg(
        #     server=PI0_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/pan_clean/g60_stovetop_zed.usd",
        #         policy=PolicyArgs(
        #             name="pi0_fast_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/block_stack_kitchen/initial_conditions_silly.json"
        #     )
        # ),

        # # pi05 jobs
        # JobCfg(
        #     server=PI05_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/block_stack_kitchen/g60_kitchen_table_zed.usd",
        #         policy=PolicyArgs(
        #             name="pi05_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/block_stack_kitchen/initial_conditions_silly.json"
        #     ),
        # ),
        # JobCfg(
        #     server=PI05_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/food_bussing/g60_corner.usd",
        #         policy=PolicyArgs(
        #             name="pi05_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/food_bussing/initial_conditions_silly.json"
        #     ),
        # ),
        # JobCfg(
        #     server=PI05_SERVER,
        #     eval_args=EvalArgs(
        #         usd="PolaRiS-assets/environments/food_bussing/g60_corner.usd",
        #         policy=PolicyArgs(
        #             name="pi05_droid_jointpos",
        #             client="DroidJointPos",
        #             open_loop_horizon=8,
        #         ),
        #         initial_conditions_file="PolaRiS-assets/environments/food_bussing/initial_conditions_silly.json"
        #     )
        # ),
    ],
)
