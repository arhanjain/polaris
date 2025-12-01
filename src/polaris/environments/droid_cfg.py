import torch
from pathlib import Path
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp
import numpy as np

from polaris.utils import DATA_PATH
from polaris.environments.robot_cfg import NVIDIA_DROID

from pxr import Usd, UsdPhysics, UsdGeom
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.managers import SceneEntityCfg 
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG

### SceneCfg ###
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    robot = NVIDIA_DROID 
    
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        # update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def __post_init__(self,):
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

    def dynamic_setup(self, environment_path, robot_splat=False, nightmare="", **kwargs):
        environment_path_ = Path(environment_path)
        environment_path = str(environment_path_.resolve())

        self.sphere_light = AssetBaseCfg(
            prim_path="/World/biglight",
            spawn=sim_utils.SphereLightCfg(intensity=3000) if nightmare else sim_utils.DomeLightCfg(intensity=1000),

            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, -0.8, 0.7)),
        )
        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=environment_path,
                    activate_contact_sensors=False,
                    ),
                )
        self.scene = scene
        stage = Usd.Stage.Open(
            environment_path
        )
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            # if rigid body
            name = child.GetName()
            print(name)
            asset = None
            if UsdPhysics.RigidBodyAPI(child):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
                asset = RigidObjectCfg(
                            prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                            spawn=None,
                            init_state=RigidObjectCfg.InitialStateCfg(
                                pos=pos,
                                rot=rot,
                            ),
                        )

            elif child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
                asset = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    height=720,
                    width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False,
                    spawn=None,
                    offset=CameraCfg.OffsetCfg(
                        pos = pos, rot = rot, convention="opengl"

                    ),
                )

            if asset:
                setattr(self, name, asset)

        # if external cam not in scene definition, we'll set a default one
        if not hasattr(self, "external_cam"):
            self.external_cam = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/external_cam",
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
                        pos = (-0.01, -0.33, 0.48), rot = (0.76, 0.43, -0.24, -0.42), convention="opengl"
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
        open_command_expr = {"finger_joint": 0.0},
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


def ee_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = math.subtract_frame_transforms(
        robot_pos_w, robot_quat_w,
        ee_pos_w, ee_quat_w
    )

    return torch.cat((ee_pos_b, ee_quat_b), dim=1)

def rigid_body_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    pos = asset.data.root_pos_w
    quat = asset.data.root_quat_w

    return torch.cat([pos, quat], dim=-1)

def reset_root_state_uniform_absolute(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    dist_from_bounds: float,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    # range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"] if key != "x" and key != "y" else pose_rangeA
    range_list = []
    # x and y
    range_list.append((pose_range["x"][0] + dist_from_bounds, pose_range["x"][1] - dist_from_bounds))
    range_list.append((pose_range["y"][0] + dist_from_bounds, pose_range["z"][1] - dist_from_bounds))
    range_list = range_list + [pose_range.get(key, (0.0, 0.0)) for key in ["z", "roll", "pitch", "yaw"]]

    ranges = torch.tensor(range_list, device=asset.device, dtype=torch.float)
    rand_samples = math.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)


    # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    positions = env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def all_joints(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.clone()
    return joint_pos

@configclass
class EnvRelCfg(EnvCfg):
    def __init__(self):
        super().__init__()
    
        self.episode_length_s = 3000

        self.actions.arm = mdp.DifferentialInverseKinematicsActionCfg(
             asset_name="robot",
             joint_names=["panda_joint.*"],
             body_name="base_link",
             controller=mdp.DifferentialIKControllerCfg(
                 command_type="pose", use_relative_mode=True, ik_method="dls"
                 ),
             body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                 pos=(0.0, 0.0, 0.0),
                 ),
        )

        self.observations.policy.ee_pose = ObsTerm(func=ee_pose) # ee frame
        self.observations.policy.all_joints = ObsTerm(func=all_joints) # all joint positions

    # for data collection randomized resets, and all rigid body pose observations
    def dynamic_setup(self, environment_path):
        super().dynamic_setup(environment_path)

        initial_conditions_path = Path(environment_path).parent.resolve() / "initial_conditions.json"

        #     initial_conditions_dict = json.load(f)
        #     pose_range = initial_conditions_dict["pose_range"]

        stage = Usd.Stage.Open(
            environment_path
        )

        # try to get the workspace bbox 
        randomization_prim = stage.GetPrimAtPath("/randomization")
        bbox_cache = UsdGeom.BBoxCache(    time=Usd.TimeCode.Default(),    includedPurposes=[UsdGeom.Tokens.default_])
        box = bbox_cache.ComputeLocalBound(randomization_prim)
        randomization_bbox =  box.ComputeAlignedBox()
        randomization_min, randomization_max = randomization_bbox.GetMin(), randomization_bbox.GetMax()
        pose_range = {
            "x": (randomization_min[0], randomization_max[0]),
            "y": (randomization_min[1], randomization_max[1]),
            "z": (randomization_min[2], randomization_max[2]),
        }

        
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()
        for child in children:
            # if rigid body
            name = child.GetName()
            child_rigid = UsdPhysics.RigidBodyAPI(child)
            if not child_rigid:
                continue

            # Observations
            obsterm = ObsTerm(
                    func = rigid_body_pose,
                    params = {
                        "asset_cfg" : SceneEntityCfg(name),
                        }
                    )
            setattr(self.observations.policy, name, obsterm)

            # Events 
            if not child_rigid.GetKinematicEnabledAttr().Get():

                bbox = bbox_cache.ComputeLocalBound(child)
                bbox = bbox.ComputeAlignedBox()
                bbox_max, bbox_min = bbox.GetMax(), bbox.GetMin()
                pos = child.GetAttribute("xformOp:translate").Get()
                longest_from_center = np.max([np.abs(bbox_max - pos), np.abs(bbox_min - pos)])

                event = EventTerm(
                        func = reset_root_state_uniform_absolute,
                        params = {
                            "asset_cfg": SceneEntityCfg(name),
                            "dist_from_bounds": longest_from_center,
                            "pose_range": {
                                **pose_range,
                                "yaw": (-np.pi, np.pi)
                                },
                            "velocity_range": {}
                            },
                        mode="reset"
                        )
                setattr(self.events, name, event)

