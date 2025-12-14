import torch
from pathlib import Path
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp
import numpy as np
from typing import Sequence

from polaris.environments.robot_cfg import NVIDIA_DROID

from pxr import Usd, UsdGeom
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG


# Patch to fix updating camera poses, since it's broken in IsaacLab 2.3
class FixedCamera(Camera):
    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # check camera prim exists
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

        # get the poses from the view
        env_ids = env_ids.to(torch.int32)
        poses, quat = self._view.get_world_poses(env_ids, usd=False)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = (
            math.convert_camera_frame_orientation_convention(
                quat, origin="opengl", target="world"
            )
        )


### SceneCfg ###
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    robot = NVIDIA_DROID

    wrist_cam = CameraCfg(
        class_type=FixedCamera,
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/biglight",
        spawn=sim_utils.DomeLightCfg(intensity=1000),
    )

    def __post_init__(
        self,
    ):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

    def dynamic_setup(self, environment_path, robot_splat=True, nightmare="", **kwargs):
        environment_path_ = Path(environment_path)
        environment_path = str(environment_path_.resolve())

        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=environment_path,
                activate_contact_sensors=False,
            ),
        )
        self.scene = scene
        if not robot_splat:
            self.robot.spawn.semantic_tags = [("class", "raytraced")]
        stage = Usd.Stage.Open(environment_path)
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            name = child.GetName()
            print(name)

            # if its a camera, use the camera pose
            if child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    height=720,
                    width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False,
                    spawn=None,
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                )
                setattr(self, name, asset)
            else:
                # All Xforms specified in scene.usda should be RigidBody
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=pos,
                        rot=rot,
                    ),
                )
                setattr(self, name, asset)

        if not hasattr(self, "external_cam"):
            self.external_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene/external_cam",
                height=720,
                width=1280,
                data_types=["rgb", "semantic_segmentation"],
                colorize_semantic_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.0476,
                    horizontal_aperture=2.5452,
                    vertical_aperture=1.4721,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(-0.01, -0.33, 0.48),
                    rot=(0.76, 0.43, -0.24, -0.42),
                    convention="opengl",
                ),
            )


### SceneCfg ###


### ActionCfg ###
class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


@configclass
class ActionCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


### ActionCfg ###


### ObsCfg ###
def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


### ObsCfg ###


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)


#### END DROID ####
