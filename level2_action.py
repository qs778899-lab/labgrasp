"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""

import numpy as np
import cv2
import os
import sys
import time
import logging
import gc
import torch
import trimesh
from typing import Any
from spatialmath import SE3, SO3
sys.path.append("FoundationPose")
from estimater import *
try:
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
except ImportError:
    pass # Assume they are available via estimater
from dino_mask import get_mask_from_GD
from qwen_mask import get_mask_from_qwen

# from calculate_grasp_pose_from_object_pose import calculate_grasp_pose_from_object_pose as choose_grasp_pose
from Utils import *
from datetime import datetime  # 在 Utils import 之后重新导入，避免被覆盖

'''
例如grasp, lift, approach, 
twist, push, align, release, pull, nudge,等
'''
 # 核心控制函数 



def detect_object_pose_using_foundation_pose(target:str,mesh_path,cam:dict[str, Any]):
    '''
    使用foundation pose来检测物体位姿
    先找到物体分割图像（grounding + sam），然后使用foundation pose来检测物体位姿

    Args:
        target: 要检测的物体
        mesh_path: 物体的mesh路径
        cam: env.camera_main (dict containing 'cam' object and 'cam_k' matrix)
    Returns:
        center_pose: 物体位姿在相机坐标
    '''

    debug = 0
    debug_dir = "debug"
    
    # Create timestamped save directory for debug images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_path)
    #? openscad的单位是mm， 但是转为obj文件后单位又变成m，所以还是需要转换！
    mesh.vertices /= 1000 #! 单位转换除以1000
    # mesh.vertices /= 3
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() 
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    
    cam_k = cam["cam_k"]
    camera = cam["cam"]

    center_pose = None

    try:
        frame_count = 0
        last_valid_pose = None  # 保存上一次有效的pose
        
        while True:
            frames = camera.get_frames()
            if frames is None:
                continue
            color = frames['color']  #get_frames获取当前帧的所有数据（RGB、深度等）
            depth = frames['depth']/1000

            color_path = os.path.join(save_dir, f"color_frame_{frame_count:06d}.png")
            print("befor foundation pose, color_shape: ", color.shape)
            cv2.imwrite(color_path, color)
            
            # 每隔15帧进行一次FoundationPose检测
            if frame_count % 15 == 0:
                mask = get_mask_from_GD(color, target)
                # mask = get_mask_from_qwen(color, target, model_path="/home/erlin/work/labgrasp/Qwen3-VL/Qwen3-VL-4B-Thinking", bbox_vis_path=os.path.join(save_dir, f"qwen_bbox_frame_{frame_count:06d}.png"))
            
                cv2.imshow("mask", mask)
                cv2.imshow("color", color)
                
                pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
                print(f"第{frame_count}帧检测完成，pose: {pose}")
                
                center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
                
                vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('object 6D pose', vis[...,::-1])
    
                mask_path = os.path.join(save_dir, f"mask_frame_{frame_count:06d}.png")
                vis_path = os.path.join(save_dir, f"vis_frame_{frame_count:06d}.png")
                cv2.imwrite(mask_path, mask)
                cv2.imwrite(vis_path, vis[...,::-1])    

                # input("break01")            

                #? 清理内存
                torch.cuda.empty_cache()
                gc.collect()
 
                last_valid_pose = center_pose  # 保存这次检测的结果
            else:
                # 使用上一次检测的结果
                center_pose = last_valid_pose
                # print(f"第{frame_count}帧使用上次检测结果")
            
            print("center_pose_object: ", center_pose) 
            
            frame_count += 1
            cv2.waitKey(1)

            if center_pose is not None:
                break

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        cv2.destroyAllWindows()

    return center_pose




def choose_grasp_pose(
    center_pose_array,
    dobot,
    T_ee_cam,
    z_xoy_angle,
    vertical_euler,
    grasp_tilt_angle,
    angle_threshold,
    T_tcp_ee_z,
    T_safe_distance,
    z_safe_distance,
    verbose=True
):
    """
    从物体位姿计算抓取姿态（不执行移动，只计算）
    
    Args:
        center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        dobot: Dobot机械臂对象
        T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象)
        z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)
        vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)
        grasp_tilt_angle: 倾斜抓取角度 (度)
        angle_threshold: z轴对齐的角度阈值 (度)
        T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米)
        T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)
        z_safe_distance: 最终移动时z方向的额外安全距离 (毫米)
        verbose: 是否打印详细信息
    
    Returns:
        grasp_pose: 抓取位置和姿态 [x, y, z, rx, ry, rz] (毫米和度)
        T_base_ee_ideal: 计算得到的理想末端执行器位姿 (SE3对象)
    """
    from scipy.spatial.transform import Rotation as R
    from grasp_utils import normalize_angle
    
    if vertical_euler is None:
        vertical_euler = [-180, 0, -90]
    
    if verbose:
        print("开始计算抓取姿态...")
    
    # ------计算在机器人基系中的object pose------
    T_cam_object = SE3(center_pose_array, check=False)
    pose_now = dobot.get_pose()  # 获取当前末端执行器位姿
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
    
    # 从当前机器人位姿构造变换矩阵 T_base_ee
    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
        np.array([x_e, y_e, z_e]) / 1000.0,  # 毫米转米
        check=False
    )
    
    # 坐标变换链: T_base_cam = T_base_ee * T_ee_cam
    T_base_cam = T_base_ee * T_ee_cam
    T_base_obj = T_base_cam * T_cam_object
    
    # ------object pose 调整------
    T_base_obj_array = np.array(T_base_obj, dtype=float)
    
    # 1. 将object pose的z轴调整为垂直桌面朝上
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:3, 2]
    target_z_axis = np.array([0, 0, 1])
    z_angle_error = np.degrees(np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)))
    
    if z_angle_error > angle_threshold:
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            rotation_matrix_new = current_rotation_matrix
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R_z_align = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            rotation_matrix_new = np.dot(R_z_align, current_rotation_matrix)
        
        T_base_obj_aligned = np.eye(4)
        T_base_obj_aligned[:3, :3] = rotation_matrix_new
        T_base_obj_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_aligned, check=False)
    else:
        T_base_obj_final = T_base_obj
    
    # 2. 将object pose的x,y轴对齐到机器人基坐标系的x,y轴
    rotation_matrix_after_z = np.array(T_base_obj_final.R)
    current_x_axis = rotation_matrix_after_z[:3, 0]
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0])
    x_projected_norm = np.linalg.norm(x_projected)
    
    if x_projected_norm > 1e-6:
        x_projected = x_projected / x_projected_norm
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        R_z_align_xy = np.array([
            [np.cos(-x_angle), -np.sin(-x_angle), 0],
            [np.sin(-x_angle), np.cos(-x_angle), 0],
            [0, 0, 1]
        ])
        rotation_matrix_final = np.dot(R_z_align_xy, rotation_matrix_after_z)
        T_base_obj_final_aligned = np.eye(4)
        T_base_obj_final_aligned[:3, :3] = rotation_matrix_final
        T_base_obj_final_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_final_aligned, check=False)
    
    # 3. 将object pose绕z轴旋转指定角度
    T_base_obj_array = T_base_obj_final.A
    current_rotation = T_base_obj_array[:3, :3]
    current_translation = T_base_obj_array[:3, 3]
    
    theta = np.radians(z_xoy_angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    new_rotation = np.dot(R_z, current_rotation)
    T_base_obj_rotated = np.eye(4)
    T_base_obj_rotated[:3, :3] = new_rotation
    T_base_obj_rotated[:3, 3] = current_translation
    T_base_obj_final = SE3(T_base_obj_rotated, check=False)
    
    # ------调整抓取姿态------
    tilted_euler = [vertical_euler[0] + grasp_tilt_angle, vertical_euler[1], vertical_euler[2]]
    
    R_target_xyz = R.from_euler('xyz', tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],
        check=False
    )
    
    # ------计算在机器人基系中，夹爪grasp即tcp的抓取姿态------
    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    
    # ------计算在机器人基系中，末端执行器ee的抓取姿态------
    T_tcp_ee = SE3(0, 0, T_tcp_ee_z)
    T_safe_distance_se3 = SE3(0, 0, T_safe_distance)
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe_distance_se3
    
    # ------提取位置和姿态------
    pos_mm = T_base_ee_ideal.t * 1000  # 转换为毫米
    rx, ry, rz = T_base_ee_ideal.rpy(unit='deg', order='zyx')
    rz = normalize_angle(rz)  # 规范化到[-180, 180]度
    
    pos_mm[2] += z_safe_distance  # 添加z方向额外安全距离
    
    grasp_pose = [pos_mm[0], pos_mm[1], pos_mm[2], rx, ry, rz]

    pre_distance = 20
    pre_grasp_pose = [pos_mm[0], pos_mm[1], pos_mm[2]+ pre_distance, rx, ry, rz]
    
    if verbose:
        print(f"计算完成 - 目标位置: [{pos_mm[0]:.2f}, {pos_mm[1]:.2f}, {pos_mm[2]:.2f}] mm")
        print(f"计算完成 - 目标姿态: rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°")
    
    return pre_grasp_pose, grasp_pose, T_base_ee_ideal


def detect_object_orientation(angle_camera, save_dir=None, max_attempts=100, verbose=True):
    """
    检测物体方向（例如玻璃棒的朝向）
    
    Args:
        angle_camera: 用于角度检测的相机对象
        save_dir: 保存检测图像的目录（可选）
        max_attempts: 最大尝试次数
        verbose: 是否打印详细信息
    
    Returns:
        avg_angle: 检测到的平均角度（度）
    """
    from calculate_grasp_pose_from_object_pose import detect_dent_orientation
    
    if verbose:
        print("\n" + "="*60)
        print("🔍 开始检测物体方向...")
        print("="*60)
    
    detected_angles = None
    avg_angle = 0.0
    detection_attempts = 0
    
    while True:
        detection_attempts += 1
        
        raw_image = angle_camera.get_current_frame()
        if raw_image is None:
            if verbose:
                print(f"第{detection_attempts}次尝试: 等待相机数据...")
            time.sleep(0.1)
            continue
        img_timestamp = time.time()
        
        if verbose:
            print(f"\n📷 第{detection_attempts}次尝试: 检测新原始图像方向 (时间戳: {img_timestamp:.2f})")
        
        detected_angles, avg_angle = detect_dent_orientation(raw_image, save_dir=save_dir)
        
        if detected_angles:
            if verbose:
                print(f"成功检测到物体朝向角度: {detected_angles}, 平均: {avg_angle:.2f}°, 绝对值: {abs(avg_angle):.2f}°")
                print("="*60)
            break
        else:
            if verbose:
                print("当前图像未检测到明显方向特征，继续等待...")
            time.sleep(0.1)
        
        # 最大尝试次数限制
        if detection_attempts >= max_attempts:
            if verbose:
                print(f"⚠️  警告: 达到最大尝试次数({max_attempts}次)，使用默认角度")
            detected_angles = []
            avg_angle = 0.0
            break
    
    return avg_angle


def adjust_object_orientation(
    dobot,
    avg_angle,
    grasp_tilt_angle,
    x_adjustment=0,
    y_adjustment=0,
    z_adjustment=-15,
    move_speed=12,
    acceleration=1,
    wait_time=9.0,
    verbose=True
):
    """
    调整物体姿态至垂直桌面向下
    
    Args:
        dobot: Dobot机械臂对象
        avg_angle: 检测到的物体平均角度（度）
        grasp_tilt_angle: 抓取时的倾斜角度（度）
        x_adjustment: x方向调整量（毫米）
        y_adjustment: y方向调整量（毫米）
        z_adjustment: z方向调整量（毫米）
        move_speed: 移动速度
        acceleration: 加速度
        wait_time: 等待时间（秒）
        verbose: 是否打印详细信息
    
    Returns:
        pose_target: 目标姿态 [x, y, z, rx, ry, rz]
        pose_after_adjust: 调整后的实际姿态 [x, y, z, rx, ry, rz]
    """
    import rospy
    
    if verbose:
        print("\n开始调整物体姿态至垂直桌面向下")

    wait_stable = rospy.Rate(1.0 / 1.0)
    wait_stable.sleep()
    
    pose_now = dobot.get_pose()
    delta_ee = abs(avg_angle) - grasp_tilt_angle
    
    # 需要让tcp朝外旋转； grasp_tilt_angle为正值时，tcp会朝外旋转。
    pose_target = [
        pose_now[0] + x_adjustment,
        pose_now[1] + y_adjustment,
        pose_now[2] + z_adjustment,
        pose_now[3] + delta_ee,
        pose_now[4],
        pose_now[5]
    ]
    
    dobot.move_to_pose(
        pose_target[0], pose_target[1], pose_target[2],
        pose_target[3], pose_target[4], pose_target[5],
        speed=move_speed,
        acceleration=acceleration
    )
    
    # 等待移动完成
    wait_rate = rospy.Rate(1.0 / wait_time)
    wait_rate.sleep()
    
    # 验证是否到达目标位置
    pose_after_adjust = dobot.get_pose()
    
    if verbose:
        print(f"姿态调整完成: Rx={pose_after_adjust[3]:.2f}° (目标: {pose_target[3]:.2f}°)")
    
    return pose_target, pose_after_adjust


def detect_contact_with_surface(
    dobot,
    contact_camera,
    save_dir,
    sample_interval=0.1,
    move_step=3,
    max_steps=700,
    change_threshold=3,
    pixel_threshold=2,
    min_area=2,
    move_speed=5,
    acceleration=1,
    verbose=True
):
    """
    检测物体是否触碰到桌面（通过相机图像变化）
    
    Args:
        dobot: Dobot机械臂对象
        contact_camera: 用于接触检测的相机对象
        save_dir: 保存调试图像的目录
        sample_interval: 采样间隔（秒）
        move_step: 每步下降距离（毫米）
        max_steps: 最大步数
        change_threshold: 变化阈值（百分比）
        pixel_threshold: 像素变化阈值
        min_area: 最小变化区域
        move_speed: 移动速度
        acceleration: 加速度
        verbose: 是否打印详细信息
    
    Returns:
        contact_detected: 是否检测到接触（布尔值）
        steps_taken: 实际下降的步数
        total_distance: 总下降距离（毫米）
    """
    import rospy
    
    if verbose:
        print("\n开始监测物体与桌面接触...")
    
    gray_debug_dir = os.path.join(save_dir, "gray_images_debug")
    os.makedirs(gray_debug_dir, exist_ok=True)
    
    rate = rospy.Rate(1.0 / sample_interval)
    rate.sleep()
    
    # 获取初始图像
    frame_before = None
    while frame_before is None:
        initial_frame = contact_camera.get_current_frame()
        if initial_frame is not None:
            frame_before = initial_frame
        else:
            if verbose:
                print("等待初始图像...")
            rospy.sleep(sample_interval)
    
    pose_current = dobot.get_pose()
    contact_detected = False
    steps_taken = 0
    
    for step in range(max_steps):
        wait = rospy.Rate(33)
        wait.sleep()
        
        # 获取动作前图像
        frame_data_before = contact_camera.get_current_frame()
        if frame_data_before is None:
            if verbose:
                print(f"  步骤 {step+1}: 等待动作前图像...")
            rospy.sleep(sample_interval)
            continue
        frame_before = frame_data_before
        
        # 向下移动一小步
        pose_current[2] -= move_step
        dobot.move_to_pose(
            pose_current[0], pose_current[1], pose_current[2],
            pose_current[3], pose_current[4], pose_current[5],
            speed=move_speed,
            acceleration=acceleration
        )
        
        # 等待并抓取动作后的新帧，连续高频采样检测
        frame_after = None
        has_change = False
        for _ in range(20):  # 0.1*20 = 2s
            rate.sleep()
            candidate_frame = contact_camera.get_current_frame()
            if candidate_frame is not None:
                frame_after = candidate_frame
                
                has_change = contact_camera.has_significant_change(
                    frame_before, frame_after,
                    change_threshold=change_threshold,
                    pixel_threshold=pixel_threshold,
                    min_area=min_area,
                    save_dir=gray_debug_dir,
                    step_num=step
                )
                
                if has_change:
                    break
        
        if frame_after is None:
            if verbose:
                print(f"  步骤 {step+1}: 未收到新图像，继续等待...")
            continue
        
        if has_change:
            contact_detected = True
            steps_taken = step + 1
            if verbose:
                print(f"检测到显著变化！物体可能已接触桌面 (步数: {steps_taken}, 下降: {steps_taken*move_step}mm)")
            break
        
        if verbose:
            print(f"  步骤 {step+1}/{max_steps}: 未检测到接触，继续下降...")
        steps_taken = step + 1
    
    if not contact_detected and verbose:
        print("达到垂直向下最大移动距离，未检测到明显变化")
    
    if verbose:
        print("下降检测完成\n")
    
    total_distance = steps_taken * move_step
    return contact_detected, steps_taken, total_distance


def execute_grasp_action(
    grasp_pose,
    dobot,
    gripper,
    gripper_close_pos,
    move_speed=8,
    gripper_force=10,
    gripper_speed=30,
    verbose=True
):
    """
    执行抓取动作
    
    Args:
        grasp_pose: 抓取位置和姿态 [x, y, z, rx, ry, rz]
        dobot: Dobot机械臂对象
        gripper: 夹爪对象
        gripper_close_pos: 夹爪闭合位置
        move_speed: 移动速度
        gripper_force: 夹爪力量
        gripper_speed: 夹爪速度
        verbose: 是否打印详细信息
    
    Returns:
        success: 是否成功执行抓取
    """
    if verbose:
        print("\n开始执行抓取动作...")
        print(f"[执行] 目标位置: [{grasp_pose[0]:.2f}, {grasp_pose[1]:.2f}, {grasp_pose[2]:.2f}] mm")
        print(f"[执行] 目标姿态: rx={grasp_pose[3]:.2f}°, ry={grasp_pose[4]:.2f}°, rz={grasp_pose[5]:.2f}°")
        print(f"[执行] 移动速度: {move_speed}")
    
    # 移动到抓取位置
    dobot.move_to_pose(
        grasp_pose[0], grasp_pose[1], grasp_pose[2],
        grasp_pose[3], grasp_pose[4], grasp_pose[5],
        speed=move_speed
    )
    
    if dobot.check_pose(grasp_pose[0], grasp_pose[1], grasp_pose[2]):
        if verbose:
            print("[执行] 到达指定抓取物体位置")
    
    # 最终位置
    final_pos = [grasp_pose[0], grasp_pose[1], grasp_pose[2]]
    dobot.move_to_pose(
        *final_pos,
        grasp_pose[3], grasp_pose[4], grasp_pose[5],
        speed=move_speed
    )
    
    if dobot.check_pose(*final_pos):
        gripper.control(gripper_close_pos, gripper_force, gripper_speed)
        print("夹爪开始闭合")
        
        # 等待夹爪到达目标位置
        timeout, interval = 5.0, 0.1
        elapsed = 0
        while elapsed < timeout:
            current = gripper.read_current_position()
            if current and abs(current[0] - gripper_close_pos) < 10:
                break
            time.sleep(interval)
            elapsed += interval
        
        if verbose:
            print("[完成] 抓取操作完成!")
        return True
    else:
        if verbose:
            print("[失败] 未能到达最终抓取位置")
        return False



  





# def grasp(
#     object, #要抓取的物体
#     arm, #机械臂
#     pre_grasp_dist, #抓取前距离
#     grap_dis, #抓取具体位置，抓取位置的z方向
#     gripper_pose#夹爪张开闭合的尺度
# ):
#     # 计算抓取姿态
#     pre_grasp_pose, grasp_pose = choose_grasp_pose()

#     action.move(pre_grasp_pose_dist)

#     action.move(grasp_pose)
#     pass
