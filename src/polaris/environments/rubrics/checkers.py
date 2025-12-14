from pxr import Usd, UsdGeom
from pxr import Gf
from omni.usd import get_context
import numpy as np
import torch


def reach(obj_name, threshold=0.05):
    """
    Returns a checker function that expects (env).
    Example: is_near("bowl", threshold=0.04)(env)
    """

    def checker(env):
        obj_pos = env.scene[obj_name].data.root_pos_w[0]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[0]
        dist = torch.norm(obj_pos - ee_pos)
        return dist < threshold

    return checker


def lift(obj_name, threshold=0.05, default_height=None):
    def checker(env):
        nonlocal default_height
        object_pos = env.scene[obj_name].data.root_pos_w[0]
        if default_height is None:
            default_height = env.scene[obj_name].data.default_root_state[0, 2]

        return (object_pos[2] - default_height).item() > threshold

    return checker


def is_within_xy(object1, object2, percent_threshold=0.5, open_finger_threshold=0.1):
    """
    Check if object1 is inside object2.
    """

    def checker(env):
        # ee should be open
        stage = get_context().get_stage()
        finger_joint = env.scene["robot"].data.joint_pos[0][
            env.scene["robot"].data.joint_names.index("finger_joint")
        ]
        if finger_joint >= open_finger_threshold:
            return False

        obj1_prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{object1}")
        obj2_prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{object2}")
        obj1_pos = env.scene[object1].data.root_pos_w[0]
        obj2_pos = env.scene[object2].data.root_pos_w[0]
        obj1_quat = env.scene[object1].data.root_quat_w[0]
        obj2_quat = env.scene[object2].data.root_quat_w[0]

        obj1_corners, obj1_centroid = get_bbox(obj1_prim, pos=obj1_pos, quat=obj1_quat)
        obj2_corners, obj2_centroid = get_bbox(obj2_prim, pos=obj2_pos, quat=obj2_quat)
        obj1_corners = np.array(obj1_corners)  # [8, 3]
        obj2_corners = np.array(obj2_corners)  # [8, 3]

        # compute intersection of xy planes
        obj1_xy_corners = obj1_corners[:, :2]  # [8, 2]
        obj2_xy_corners = obj2_corners[:, :2]  # [8, 2]
        obj1_min_xy = np.min(obj1_xy_corners, axis=0)
        obj1_max_xy = np.max(obj1_xy_corners, axis=0)
        obj2_min_xy = np.min(obj2_xy_corners, axis=0)
        obj2_max_xy = np.max(obj2_xy_corners, axis=0)

        # Overlap rectangle boundaries
        overlap_min_x = max(obj1_min_xy[0], obj2_min_xy[0])
        overlap_max_x = min(obj1_max_xy[0], obj2_max_xy[0])
        overlap_min_y = max(obj1_min_xy[1], obj2_min_xy[1])
        overlap_max_y = min(obj1_max_xy[1], obj2_max_xy[1])

        # Check if there's any actual overlap
        if overlap_min_x >= overlap_max_x or overlap_min_y >= overlap_max_y:
            return False

        # Areas
        obj1_area = (obj1_max_xy[0] - obj1_min_xy[0]) * (
            obj1_max_xy[1] - obj1_min_xy[1]
        )
        overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)

        # Percentage of object1 area that is inside object2
        overlap_ratio = overlap_area / obj1_area
        # print(f"{object1} is inside {object2} {overlap_ratio}")

        return overlap_ratio >= percent_threshold

    return checker


def get_scale(prim: Usd.Prim) -> Gf.Vec3d:
    """
    Get the scale parameter applied to a Usd.Prim.

    This function tries multiple approaches to get the scale:
    1. Directly from the 'xformOp:scale' attribute if it exists
    2. From the transform matrix using ExtractScale() method
    3. Returns (1, 1, 1) as default if no scale is found

    Args:
        prim: The Usd.Prim to get scale from

    Returns:
        Gf.Vec3d: The scale vector (x, y, z)
    """
    # First try to get scale directly from xformOp:scale attribute
    # scale_attr = get_attribute(prim, "xformOp:scale")
    scale_attr = prim.GetAttribute("xformOp:scale")
    if scale_attr and scale_attr.IsValid():
        scale_value = scale_attr.Get()
        if scale_value is not None:
            # Convert to Gf.Vec3d if it's not already
            if isinstance(scale_value, (list, tuple)):
                return Gf.Vec3d(*scale_value)
            elif hasattr(scale_value, "__len__") and len(scale_value) == 3:
                return Gf.Vec3d(scale_value[0], scale_value[1], scale_value[2])
            else:
                return Gf.Vec3d(scale_value, scale_value, scale_value)

    # Default scale if nothing else works
    return Gf.Vec3d(1.0, 1.0, 1.0)


def get_bbox(body_prim: Usd.Prim, pos=None, quat=None, scalar_first=False):
    pos = pos.cpu().numpy().astype(np.float64)
    quat = quat.cpu().numpy().astype(np.float64)
    ## TODO: add options: zero_centered, current
    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(
        time_code, includedPurposes=[UsdGeom.Tokens.default_]
    )
    bbox_cache.Clear()

    prim_bbox = bbox_cache.ComputeLocalBound(body_prim)

    # Get corners and centroid # corners = prim_bbox.GetCorners()      # List of 8 GfVec3d points [1][2]
    range3d = prim_bbox.GetRange()
    matrix = prim_bbox.GetMatrix()

    corners = [matrix.Transform(range3d.GetCorner(i)) for i in range(8)]
    centroid = prim_bbox.ComputeCentroid()  # GfVec3d [1][2]

    # Transform to origin using the inverse of the prim's world transform
    xform_cache = UsdGeom.XformCache(time_code)
    world_xform = xform_cache.GetLocalToWorldTransform(body_prim)  # Gf.Matrix4d[2][3]
    transform = world_xform.GetInverse()

    transformed_corners = [transform.Transform(corner) for corner in corners]
    transformed_centroid = transform.Transform(centroid)

    scale = get_scale(body_prim)
    if scale is not Gf.Vec3d(1.0, 1.0, 1.0):
        # Scale corners directly
        scaled_corners = []
        for corner in transformed_corners:
            # Scale each corner directly
            scaled_corner = Gf.Vec3d(
                corner[0] * scale[0], corner[1] * scale[1], corner[2] * scale[2]
            )
            scaled_corners.append(scaled_corner)

        transformed_corners = scaled_corners

    # User-supplied transform, if any
    if quat is not None and pos is not None:
        if isinstance(pos, np.ndarray) or isinstance(pos, list):
            # pos = np_to_gf_vec3d(pos)
            pos = Gf.Vec3d(pos[0], pos[1], pos[2])
        if isinstance(quat, np.ndarray) or isinstance(quat, list):
            # quat = np_to_gf_quatf(quat, scalar_first)
            quat = Gf.Quatd(quat[0], quat[1], quat[2], quat[3])

        additional_transform = Gf.Matrix4d().SetRotateOnly(quat)
        additional_transform.SetTranslateOnly(pos)
    else:
        # No additional transform
        return transformed_corners, transformed_centroid

    # Apply additional transform
    new_corners = [
        additional_transform.Transform(corner) for corner in transformed_corners
    ]
    new_centroid = additional_transform.Transform(transformed_centroid)

    return new_corners, new_centroid
