#!/usr/bin/env python3
import sys
import signal
import atexit
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD 
from qwen_mask import get_mask_from_qwen
from create_camera import CreateRealsense
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
import json
import threading
from datetime import datetime
import gc
import torch
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
###from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose, detect_dent_orientation
from camera_reader import CameraReader
from level2_action import detect_object_pose_using_foundation_pose, choose_grasp_pose, execute_grasp_action, detect_object_orientation, adjust_object_orientation, detect_contact_with_surface
from env import create_env
import rospy
from std_msgs.msg import Float64MultiArray


def _lerp_angle_deg(start: float, end: float, ratio: float) -> float:
    """
    Linearly interpolate two angles expressed in degrees while keeping the
    result inside [-180, 180] to avoid discontinuities.
    """
    delta = normalize_angle(end - start)
    return normalize_angle(start + delta * ratio)


def _lerp_pose(start_pose: np.ndarray, end_pose: np.ndarray, ratio: float) -> np.ndarray:
    """Interpolate position linearly and orientation via shortest angular path."""
    translation = start_pose[:3] + (end_pose[:3] - start_pose[:3]) * ratio
    rotations = [
        _lerp_angle_deg(start_pose[3 + i], end_pose[3 + i], ratio) for i in range(3)
    ]
    return np.concatenate([translation, np.array(rotations, dtype=float)])


def interpolate_dobot_path(anchor_poses, target_count=20, direction="ccw"):
    """
    Generate densely sampled poses along a closed path defined by anchor poses.

    Args:
        anchor_poses: Iterable of [x, y, z, rx, ry, rz] points ordered along the path.
        target_count: Total number of poses to return (duplicates of the final point
                      are avoided by sampling on [0, perimeter)).
        direction: "ccw" keeps the provided order; "cw" reverses it.

    Returns:
        List of interpolated poses following the requested direction.
    """
    if len(anchor_poses) < 2:
        raise ValueError("At least two anchor poses are required.")

    direction = direction.lower()
    if direction not in {"ccw", "cw"}:
        raise ValueError("direction must be either 'ccw' or 'cw'.")

    ordered = anchor_poses if direction == "ccw" else list(reversed(anchor_poses))
    waypoints = np.asarray(ordered, dtype=float)

    # Build a closed loop in Cartesian space (only xyz influences arc-length).
    closed_xyz = np.vstack([waypoints[:, :3], waypoints[0, :3]])
    segment_lengths = np.linalg.norm(np.diff(closed_xyz, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    if np.isclose(total_length, 0.0):
        raise ValueError("Anchor poses collapse to a single point; cannot interpolate.")

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_distances = np.linspace(0.0, cumulative[-1], target_count, endpoint=False)

    interpolated = []
    for distance_along_path in sample_distances:
        seg_idx = np.searchsorted(cumulative, distance_along_path, side="right") - 1
        seg_idx = min(seg_idx, len(segment_lengths) - 1)
        seg_length = segment_lengths[seg_idx]
        if seg_length == 0:
            ratio = 0.0
        else:
            ratio = (distance_along_path - cumulative[seg_idx]) / seg_length

        start_pose = waypoints[seg_idx]
        end_pose = waypoints[(seg_idx + 1) % len(waypoints)]
        interpolated.append(_lerp_pose(start_pose, end_pose, ratio).tolist())

    return interpolated


camera = None
angle_camera = None
contact_camera = None
dobot = None
gripper = None
preview_running = None



if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"图像将保存到: {save_dir}")
    angle_log_path = os.path.join(save_dir, "angle_log.csv")
    with open(angle_log_path, 'w') as f:
        f.write("frame,timestamp,angle_z_deg,detected_angles,avg_angle\n")
    # print(f"角度数据将保存到: {angle_log_path}")

    # 使用 env.py 初始化环境（包含机械臂、相机等）
    env = create_env("config.json")
    robot_main = env.robot1
    dobot = robot_main["robot"]
    gripper = env.gripper
    camera_main = env.camera1_main
    camera = camera_main["cam"]  # 为清理函数设置全局变量
    T_ee_cam = camera_main["T_ee_cam"]
    
    # 从 GraspLibrary.json 加载抓取参数
    target_object = "stirring rod"  # 可以修改为: "red cylinder", "red stirring rod", "stirring rod"
    with open("GraspLibrary.json", 'r') as f:
        grasp_library = json.load(f)
    if target_object not in grasp_library:
        raise ValueError(f"目标物体 '{target_object}' 不在 GraspLibrary.json 中")
    grasp_params = grasp_library[target_object]
    print(f"\n加载目标物体: {target_object}")
    print(f"抓取参数: {grasp_params}\n")
    
    # mesh_file = "mesh/cube.obj"
    # mesh_file = "mesh/thin_cube.obj"
    mesh_file = "mesh/cube_1_20.obj"
    key = cv2.waitKey(1)
 
    # 从 GraspLibrary 获取抓取参数
    z_xoy_angle = grasp_params["z_xoy_angle"]
    vertical_euler = grasp_params["vertical_euler"]
    grasp_tilt_angle = grasp_params["grasp_tilt_angle"]
    angle_threshold = grasp_params["angle_threshold"]
    T_safe_distance = grasp_params["T_safe_distance"]
    z_safe_distance = grasp_params["z_safe_distance"]
    gripper_close_pos = grasp_params["gripper_close_pos"]

    # 手动标定的基准姿态（请根据实测顺序排列，可替换为 6 个或更多点）
    anchor_poses = [
        [616, -170, 540, -178, 0, -92],
        [558, 5, 540, -178, -5, -92],
        [460, 100, 540, -172, -10, -92],
        [368, 5, 540, -172, -5, -92],
        [317, -170, 540, -170, -2, -92],
        [368, -370, 540, -172, 5, -92],
        [460, -490, 540, -172, 10, -92],
        [558, -370, 540, -172, 5, -92],
    ]

    total_waypoints = 20
    path_direction = "ccw"  # 改成 "cw" 可按顺时针顺序生成
    interpolated_waypoints = interpolate_dobot_path(
        anchor_poses, target_count=total_waypoints, direction=path_direction
    )

    print(f"Generated {total_waypoints} waypoints ({path_direction}):")
    for idx, pose in enumerate(interpolated_waypoints, start=1):
        rounded_pose = [round(value, 3) for value in pose]
        print(f"{idx:02d}: {rounded_pose}")

    for pose in interpolated_waypoints:
        wait = rospy.Rate(1.0 / 6)
        wait.sleep()
        dobot.move_to_pose(pose)  
