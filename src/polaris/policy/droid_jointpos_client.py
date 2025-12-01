import tyro
import numpy as np
from PIL import Image
from dataclasses import dataclass
# from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient, PolicyArgs



# Joint Position Client for DROID
@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs ) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointPosClient")


        # self.client = websocket_client_policy.WebsocketClientPolicy(
        #     args.remote_host, args.remote_port
        # )

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon

    @property
    def rerender(self) -> bool:
        return self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.args.open_loop_horizon

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self, **kwargs):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        both = None
        ret = {}
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.args.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)

            self.actions_from_chunk_completed = 0

            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            # self.pred_action_chunk = self.client.infer(request_data)["actions"]
            server_response= self.client.infer(request_data)
            self.pred_action_chunk = server_response["actions"]
            ret["server_timing"] = server_response["server_timing"]

        # assert self.pred_action_chunk.shape == (10, 8)
        curr_obs = self._extract_observation(obs)
        both = np.concatenate([
            image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
            image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
        ], axis=1)
        ret["viz"] = both

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        ret["action"] = action
        return ret

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


RELATIVE_MAX_JOINT_DELTA = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
MAX_JOINT_DELTA = RELATIVE_MAX_JOINT_DELTA.max()


def joint_delta_to_velocity(joint_delta):
    if isinstance(joint_delta, list):
        joint_delta = np.array(joint_delta)

    return joint_delta / RELATIVE_MAX_JOINT_DELTA


def joint_velocity_to_delta(joint_velocity):
    if isinstance(joint_velocity, list):
        joint_velocity = np.array(joint_velocity)

    relative_max_joint_vel = joint_delta_to_velocity(RELATIVE_MAX_JOINT_DELTA)
    max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()

    if max_joint_vel_norm > 1:
        joint_velocity = joint_velocity / max_joint_vel_norm

    joint_delta = joint_velocity * MAX_JOINT_DELTA

    return joint_delta


if __name__ == "__main__":
    import torch
    args = tyro.cli(Args)
    client = Client(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
