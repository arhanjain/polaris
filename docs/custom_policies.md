# Cotraining 
Pull the RLDS sim co-training dataset 
```bash
uvx hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir [path/to/rlds/datasets]
```

To run cotraining on an off-the-shelf policy, use the PolaRiS training configs in [openpi](https://github.com/Physical-Intelligence/openpi). Before running make sure to update the `rlds_data_dir` for each config. Example run below.
```bash
cd third_party/openpi
uv run --group rlds scripts/train.py  pi05_droid_jointpos_polaris --exp-name=polaris-pi05-droid --overwrite
```


# Evaluating Custom Policies

PolaRiS provides a simple interface for evaluating custom policies. For simplicity, we employ a server-client setup where the policy is hosted in a different process from the evaluation process. This can be helpful especialy when policies may be require lots of resources or conflicting dependencies. 

We interface with policies through [openpi's WebsockeClientPolicy](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py). You may host the policy server however you want. To define a client you need to implement the [InferenceClient](src/polaris/policy/abstract_client.py) abstract class. See [DroidJointPosClient](src/polaris/policy/droid_jointpos_client.py) for a working example. 

Minimal Example:
```py
@InferenceClient.register(client_name="CustomPolicy")
class CustomPolicy(InferenceClient):
    def __init__(self, args: PolicyArgs):
        # inititalize any necessary state (obs history, action chunks, etc.)
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )

    @property
    def rerender(self) -> bool:
        """
        Policy requests a rerender of the visualization. Optimization for less splat rendering
        for chunked policies. Can default to always True if optimization is not desired.
        """
        return True

    def infer(self, obs, instruction, return_viz: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Does inference on observation and returns action and visualization. If visualization is not needed, return None.
        """
        request_data = {
            "external_image": obs["splat"]["external_cam"],
            "wrist_image": obs["splat"]["wrist_cam"],
            "instruction": instruction,
        }
        server_response= self.client.infer(request_data)
        return server_response["action"], None 

    def reset(self):
        """
        Resets the client to start a new episode. Useful if policy is stateful.
        """
        pass
```

To run the policy, specify the client name and port:
```bash
uv run scripts/eval.py --environment DROID-FoodBussing --policy.client CustomPolicy --policy.port 8000 --run-folder runs/test
```
