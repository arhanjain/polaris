from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

@dataclass
class PolicyArgs:
    name: str # name of the policy (pi05_droid_jointpos, pi0_fast_droid_jointpos, etc.)
    client: str # name of the client (DroidJointPos, Fake, etc.)
    host: str = "0.0.0.0"
    port: int = 8000
    open_loop_horizon: int | None = None


class InferenceClient(ABC):
    REGISTERED_CLIENTS = {}

    # def __init_subclass__(cls, client_name: str, *args, **kwargs) -> None:
    #     super().__init_subclass__(*args, **kwargs)
    #     InferenceClient.REGISTERED_CLIENTS[client_name] = cls

    @staticmethod
    def register(client_name: str) -> Callable[[type], type]:
        def decorator(cls: type):
            InferenceClient.REGISTERED_CLIENTS[client_name] = cls
            return cls
        return decorator

    @staticmethod
    def get_client(policy_args: PolicyArgs) -> 'InferenceClient':
        if policy_args.client not in InferenceClient.REGISTERED_CLIENTS:
            raise ValueError(f"Client {policy_args.client} not found. Available clients: {list(InferenceClient.REGISTERED_CLIENTS.keys())}")
        return InferenceClient.REGISTERED_CLIENTS[policy_args.client].__init__(policy_args)

    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @abstractmethod
    def rerender(self) -> bool:
        """
        Policy requests a rerender of the visualization. Optimization for less splat rendering
        for chunked policies. Can default to always True if optimization is not desired.
        """
        return True

    @abstractmethod
    def infer(self, obs, instruction) -> dict:
        """
        Does inference on observation and returns action and visualization.
        """

        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode. Useful if policy is stateful.
        """
        pass

class FakeClient(InferenceClient):
    '''
    Fake client that returns a dummy action and visualization.
    '''
    def __init__(self, *args, **kwargs) -> None:
        return

    def infer(self, obs, instruction) -> dict:
        import numpy as np
        import cv2

        external = obs["splat"]["external_cam"]
        wrist = obs["splat"]["wrist_cam"]
        external = cv2.resize(external, (224, 224))
        wrist = cv2.resize(wrist, (224, 224))
        both = np.concatenate([external, wrist], axis=1)
        return {
            "action": np.zeros((8,)),
            "viz": both,
        }

    def reset(self, *args, **kwargs):
        return