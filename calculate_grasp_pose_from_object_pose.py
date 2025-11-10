import cv2
import numpy as np
import time
import rospy
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, print_pose_info
import matplotlib.pyplot as plt
# from dir_detection import preprocessing



def execute_grasp_from_object_pose(
    center_pose_array,
    dobot,
    gripper,
    T_ee_cam,
    z_xoy_angle,
    vertical_euler,
    grasp_tilt_angle,
    angle_threshold,
    T_tcp_ee_z,
    T_safe_distance,
    z_safe_distance,
    gripper_close_pos,
    enable_gripper=True,
    verbose=True
):
    """
    从物体位姿计算并执行抓取动作
    
    Args:
        center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        dobot: Dobot机械臂对象
        gripper: 夹爪对象
        T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象)
        z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)
        vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)，默认[-180, 0, -90]
        grasp_tilt_angle: 倾斜抓取角度 (度)，叠加在vertical_euler[0]上, 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度.
        angle_threshold: z轴对齐的角度阈值 (度)
        T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米)
        T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)
        z_safe_distance: 最终移动时z方向的额外安全距离,也是为了抓取物体靠上的部分。 (毫米)
        gripper_close_pos: 夹爪闭合位置 (0-1000)，默认80
        enable_gripper: 是否执行夹爪抓取动作，默认True
        verbose: 是否打印详细信息
    
    Returns:
        success: 是否成功执行抓取
        T_base_ee_ideal: 计算得到的理想末端执行器位姿 (SE3对象)
    """
    
    # 硬编码的控制参数
    move_speed = 8
    gripper_open_pos = 1000
    gripper_force = 10
    gripper_speed = 30
    
    if vertical_euler is None:
        vertical_euler = [-180, 0, -90]
    
    if verbose:
        print("开始执行本次抓取或者移动操作...")
        # print("="*60)
    
    # # ------check part : 打印物体在相机坐标系中的位姿------
    # if verbose:
    #     print("\n[检查] 物体在相机坐标系中的位姿:")
    #     print_pose_info(center_pose_array, "物体位姿 (相机坐标系)")
    
    # ------计算在机器人基系中的object pose------
    T_cam_object = SE3(center_pose_array, check=False)
    pose_now = dobot.get_pose()  # 获取当前末端执行器位姿
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
    
    # if verbose:
    #     print(f"\n[信息] 当前机器人位姿: {pose_now}")
    
    # 从当前机器人位姿构造变换矩阵 T_base_ee
    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
        np.array([x_e, y_e, z_e]) / 1000.0,  # 毫米转米
        check=False
    )
    
    # 坐标变换链: T_base_cam = T_base_ee * T_ee_cam
    T_base_cam = T_base_ee * T_ee_cam
    # T_base_obj = T_base_cam * T_cam_obj（物体在机器人基坐标系中的位姿）
    T_base_obj = T_base_cam * T_cam_object
    
    # ------check part : 检查在机器人基系中，object pose的z轴方向------
    # if verbose:
    #     print("\n[检查] 物体在机器人基坐标系中的位姿:")
    #     print_pose_info(T_base_obj, "物体位姿 (机器人基坐标系)")
    
    # ------object pose 调整------
    T_base_obj_array = np.array(T_base_obj, dtype=float)
    
    # 1. 将object pose的z轴调整为垂直桌面朝上
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:3, 2]  # 提取当前z轴方向
    target_z_axis = np.array([0, 0, 1])  # 目标z轴方向（垂直向上）
    # 计算当前z轴与目标z轴的夹角
    z_angle_error = np.degrees(np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)))
    
    if verbose:
        pass
        # print(f"\n[姿态调整] 当前z轴与垂直方向的偏差: {z_angle_error:.2f}°")
    
    if z_angle_error > angle_threshold:
        if verbose:
            pass
            # print("z轴偏差较大，进行对齐...")
        
        # 计算旋转轴（两向量叉乘）
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:  # 两轴几乎平行
            rotation_matrix_new = current_rotation_matrix
        else:
            rotation_axis = rotation_axis / rotation_axis_norm  # 单位化旋转轴
            rotation_angle = np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
            # 构造反对称矩阵K（用于Rodrigues旋转公式）
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            # Rodrigues旋转公式: R = I + sin(θ)K + (1-cos(θ))K²
            R_z_align = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            rotation_matrix_new = np.dot(R_z_align, current_rotation_matrix)
        
        T_base_obj_aligned = np.eye(4)
        T_base_obj_aligned[:3, :3] = rotation_matrix_new
        T_base_obj_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_aligned, check=False)
    else:
        if verbose:
            pass
            # print("z轴偏差在可接受范围内，使用原始姿态")
        T_base_obj_final = T_base_obj
    
    # 2. 将object pose的x,y轴对齐到机器人基坐标系的x,y轴
    rotation_matrix_after_z = np.array(T_base_obj_final.R)
    current_x_axis = rotation_matrix_after_z[:3, 0]  # 提取当前x轴方向
    # 将x轴投影到水平面（xy平面）
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0])
    x_projected_norm = np.linalg.norm(x_projected)
    
    if x_projected_norm > 1e-6:
        x_projected = x_projected / x_projected_norm  # 单位化投影向量
        # 计算投影与基坐标系x轴的夹角
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        # 构造绕z轴旋转矩阵（消除该夹角）
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
        if verbose:
            pass
            # print("x,y轴对齐完成")
    else:
        if verbose:
            pass
            # print("x轴在水平面的投影太小，跳过x,y轴对齐")
    
    # 3. 将object pose绕z轴旋转指定角度
    # if verbose:
    #     print(f"\n[姿态调整] 将object pose绕z轴旋转{z_xoy_angle}度...")
    
    T_base_obj_array = T_base_obj_final.A
    current_rotation = T_base_obj_array[:3, :3]
    current_translation = T_base_obj_array[:3, 3]
    
    # 构造绕z轴旋转的旋转矩阵
    theta = np.radians(z_xoy_angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    new_rotation = np.dot(R_z, current_rotation)  # 左乘以在基坐标系中旋转
    T_base_obj_rotated = np.eye(4)
    T_base_obj_rotated[:3, :3] = new_rotation
    T_base_obj_rotated[:3, 3] = current_translation
    T_base_obj_final = SE3(T_base_obj_rotated, check=False)
    
    # ------check part : 检查调整后的物体姿态------
    # if verbose:
    #     print("\n[检查] 调整后的最终物体姿态:")
    #     print_pose_info(T_base_obj_final, "调整后物体位姿 (机器人基坐标系)")
    
    # ------调整抓取姿态------
    # 在垂直抓取基础上叠加倾斜角度
    tilted_euler = [vertical_euler[0] + grasp_tilt_angle, vertical_euler[1], vertical_euler[2]]
    
    # if verbose:
    #     print(f"\n[抓取姿态] 垂直抓取姿态: {vertical_euler}")
    #     print(f"[抓取姿态] 倾斜角度: {grasp_tilt_angle}°")
    #     print(f"[抓取姿态] 最终抓取姿态: {tilted_euler}")
    
    # 从欧拉角构造抓取姿态（相对于物体坐标系）
    R_target_xyz = R.from_euler('xyz', tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],  # 抓取点在物体中心
        check=False
    )
    
    # ------check part : 检查相对抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 相对抓取姿态 (物体坐标系):")
        print_pose_info(T_object_grasp_ideal, "T_object_grasp_ideal")
    
    # ------计算在机器人基系中，夹爪grasp即tcp的抓取姿态------
    # 坐标变换链: T_base_grasp = T_base_obj * T_obj_grasp
    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    
    # ------check part : 检查最终抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 最终抓取姿态 (机器人基坐标系):")
        print_pose_info(T_base_grasp_ideal, "T_base_grasp_ideal")
    
    # ------计算在机器人基系中，末端执行器ee的抓取姿态------
    # TCP到末端执行器的偏移（z方向）
    T_tcp_ee = SE3(0, 0, T_tcp_ee_z)
    T_safe_distance = SE3(0, 0, T_safe_distance)  # 额外安全距离
    # 变换链: T_base_ee = T_base_grasp * T_grasp_tcp * T_tcp_ee * T_safe
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe_distance
    
    # ------执行抓取动作------
    pos_mm = T_base_ee_ideal.t * 1000  # 转换为毫米
    # 提取ZYX欧拉角（机械臂使用的旋转顺序）
    rx, ry, rz = T_base_ee_ideal.rpy(unit='deg', order='zyx')
    rz = normalize_angle(rz)  # 规范化到[-180, 180]度
    
    pos_mm[2] += z_safe_distance  # 添加z方向额外安全距离（避免碰撞）
    
    if verbose:
        print(f"\n[执行] 目标位置: [{pos_mm[0]:.2f}, {pos_mm[1]:.2f}, {pos_mm[2]:.2f}] mm")
        print(f"[执行] 目标姿态: rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°")
        print(f"[执行] 移动速度: {move_speed}")
    
    adjusted_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]
    
    # 移动到抓取位置
    dobot.move_to_pose(*adjusted_pos, rx, ry, rz, speed=move_speed)
    
    if dobot.check_pose(*adjusted_pos):
        if verbose:
            print("[执行] 到达指定抓取物体位置")
        if enable_gripper:
            gripper.control(gripper_open_pos, gripper_force, gripper_speed)
    
    # 最终位置（要不要去掉安全距离）
    final_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]
    dobot.move_to_pose(*final_pos, rx, ry, rz, speed=move_speed)
    
    if dobot.check_pose(*final_pos):
        if verbose:
            pass
            # print("[执行] 再次确认到达抓取位置，执行抓取")
        gripper.control(gripper_close_pos, gripper_force, gripper_speed)
        
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
            pass
            # print("[完成] 抓取操作完成!")
        return True, T_base_ee_ideal
    else:
        if verbose:
            pass
            # print("[失败] 未能到达最终抓取位置")
        return False, T_base_ee_ideal





#------dobot_pose设置为固定值，和record时的pose一致------
def calculate_grasppose_from_objectpose(
    center_pose_array,
    dobot,
    gripper,
    T_ee_cam,
    z_xoy_angle,
    vertical_euler,
    grasp_tilt_angle,
    angle_threshold,
    T_tcp_ee_z,
    T_safe_distance,
    z_safe_distance,
    gripper_close_pos,
    enable_gripper=True,
    verbose=True
):
    """
    从物体位姿计算并执行抓取动作
    
    Args:
        center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        dobot: Dobot机械臂对象
        gripper: 夹爪对象
        T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象)
        z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)
        vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)，默认[-180, 0, -90]
        grasp_tilt_angle: 倾斜抓取角度 (度)，叠加在vertical_euler[0]上, 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度.
        angle_threshold: z轴对齐的角度阈值 (度)
        T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米)
        T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)
        z_safe_distance: 最终移动时z方向的额外安全距离,也是为了抓取物体靠上的部分。 (毫米)
        gripper_close_pos: 夹爪闭合位置 (0-1000)，默认80
        enable_gripper: 是否执行夹爪抓取动作，默认True
        verbose: 是否打印详细信息
    
    Returns:
        success: 是否成功执行抓取
        T_base_ee_ideal: 计算得到的理想末端执行器位姿 (SE3对象)
    """
    
    # 硬编码的控制参数
    move_speed = 5
    gripper_open_pos = 1000
    gripper_force = 10
    gripper_speed = 20
    
    if vertical_euler is None:
        vertical_euler = [-180, 0, -90]
    
    if verbose:
        print("开始执行本次抓取或者移动操作...")
        # print("="*60)
    
    # # ------check part : 打印物体在相机坐标系中的位姿------
    # if verbose:
    #     print("\n[检查] 物体在相机坐标系中的位姿:")
    #     print_pose_info(center_pose_array, "物体位姿 (相机坐标系)")
    
    # ------计算在机器人基系中的object pose------
    T_cam_object = SE3(center_pose_array, check=False)
    # pose_now = dobot.get_pose()  # 获取当前末端执行器位姿
    pose_now = [470, -20, 430, 195, 0, -90]
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
    
    # if verbose:
    #     print(f"\n[信息] 当前机器人位姿: {pose_now}")
    
    # 从当前机器人位姿构造变换矩阵 T_base_ee
    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
        np.array([x_e, y_e, z_e]) / 1000.0,  # 毫米转米
        check=False
    )
    
    # 坐标变换链: T_base_cam = T_base_ee * T_ee_cam
    T_base_cam = T_base_ee * T_ee_cam
    # T_base_obj = T_base_cam * T_cam_obj（物体在机器人基坐标系中的位姿）
    T_base_obj = T_base_cam * T_cam_object
    
    # ------check part : 检查在机器人基系中，object pose的z轴方向------
    # if verbose:
    #     print("\n[检查] 物体在机器人基坐标系中的位姿:")
    #     print_pose_info(T_base_obj, "物体位姿 (机器人基坐标系)")
    
    # ------object pose 调整------
    T_base_obj_array = np.array(T_base_obj, dtype=float)
    
    # 1. 将object pose的z轴调整为垂直桌面朝上
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:3, 2]  # 提取当前z轴方向
    target_z_axis = np.array([0, 0, 1])  # 目标z轴方向（垂直向上）
    # 计算当前z轴与目标z轴的夹角
    z_angle_error = np.degrees(np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)))
    
    if verbose:
        pass
        # print(f"\n[姿态调整] 当前z轴与垂直方向的偏差: {z_angle_error:.2f}°")
    
    if z_angle_error > angle_threshold:
        if verbose:
            pass
            # print("z轴偏差较大，进行对齐...")
        
        # 计算旋转轴（两向量叉乘）
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:  # 两轴几乎平行
            rotation_matrix_new = current_rotation_matrix
        else:
            rotation_axis = rotation_axis / rotation_axis_norm  # 单位化旋转轴
            rotation_angle = np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
            # 构造反对称矩阵K（用于Rodrigues旋转公式）
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            # Rodrigues旋转公式: R = I + sin(θ)K + (1-cos(θ))K²
            R_z_align = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            rotation_matrix_new = np.dot(R_z_align, current_rotation_matrix)
        
        T_base_obj_aligned = np.eye(4)
        T_base_obj_aligned[:3, :3] = rotation_matrix_new
        T_base_obj_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_aligned, check=False)
    else:
        if verbose:
            pass
            # print("z轴偏差在可接受范围内，使用原始姿态")
        T_base_obj_final = T_base_obj
    
    # 2. 将object pose的x,y轴对齐到机器人基坐标系的x,y轴
    rotation_matrix_after_z = np.array(T_base_obj_final.R)
    current_x_axis = rotation_matrix_after_z[:3, 0]  # 提取当前x轴方向
    # 将x轴投影到水平面（xy平面）
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0])
    x_projected_norm = np.linalg.norm(x_projected)
    
    if x_projected_norm > 1e-6:
        x_projected = x_projected / x_projected_norm  # 单位化投影向量
        # 计算投影与基坐标系x轴的夹角
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        # 构造绕z轴旋转矩阵（消除该夹角）
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
        if verbose:
            pass
            # print("x,y轴对齐完成")
    else:
        if verbose:
            pass
            # print("x轴在水平面的投影太小，跳过x,y轴对齐")
    
    # 3. 将object pose绕z轴旋转指定角度
    # if verbose:
    #     print(f"\n[姿态调整] 将object pose绕z轴旋转{z_xoy_angle}度...")
    
    T_base_obj_array = T_base_obj_final.A
    current_rotation = T_base_obj_array[:3, :3]
    current_translation = T_base_obj_array[:3, 3]
    
    # 构造绕z轴旋转的旋转矩阵
    theta = np.radians(z_xoy_angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    new_rotation = np.dot(R_z, current_rotation)  # 左乘以在基坐标系中旋转
    T_base_obj_rotated = np.eye(4)
    T_base_obj_rotated[:3, :3] = new_rotation
    T_base_obj_rotated[:3, 3] = current_translation
    T_base_obj_final = SE3(T_base_obj_rotated, check=False)
    
    # ------check part : 检查调整后的物体姿态------
    # if verbose:
    #     print("\n[检查] 调整后的最终物体姿态:")
    #     print_pose_info(T_base_obj_final, "调整后物体位姿 (机器人基坐标系)")
    
    # ------调整抓取姿态------
    # 在垂直抓取基础上叠加倾斜角度
    tilted_euler = [vertical_euler[0] + grasp_tilt_angle, vertical_euler[1], vertical_euler[2]]
    
    # if verbose:
    #     print(f"\n[抓取姿态] 垂直抓取姿态: {vertical_euler}")
    #     print(f"[抓取姿态] 倾斜角度: {grasp_tilt_angle}°")
    #     print(f"[抓取姿态] 最终抓取姿态: {tilted_euler}")
    
    # 从欧拉角构造抓取姿态（相对于物体坐标系）
    R_target_xyz = R.from_euler('xyz', tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],  # 抓取点在物体中心
        check=False
    )
    
    # ------check part : 检查相对抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 相对抓取姿态 (物体坐标系):")
        print_pose_info(T_object_grasp_ideal, "T_object_grasp_ideal")
    
    # ------计算在机器人基系中，夹爪grasp即tcp的抓取姿态------
    # 坐标变换链: T_base_grasp = T_base_obj * T_obj_grasp
    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    
    # ------check part : 检查最终抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 最终抓取姿态 (机器人基坐标系):")
        print_pose_info(T_base_grasp_ideal, "T_base_grasp_ideal")
    
    # ------计算在机器人基系中，末端执行器ee的抓取姿态------
    # TCP到末端执行器的偏移（z方向）
    T_tcp_ee = SE3(0, 0, T_tcp_ee_z)
    T_safe_distance = SE3(0, 0, T_safe_distance)  # 额外安全距离
    # 变换链: T_base_ee = T_base_grasp * T_grasp_tcp * T_tcp_ee * T_safe
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe_distance
    
    # ------执行抓取动作------
    pos_mm = T_base_ee_ideal.t * 1000  # 转换为毫米
    # 提取ZYX欧拉角（机械臂使用的旋转顺序）
    rx, ry, rz = T_base_ee_ideal.rpy(unit='deg', order='zyx')
    rz = normalize_angle(rz)  # 规范化到[-180, 180]度
    
    pos_mm[2] += z_safe_distance  # 添加z方向额外安全距离（避免碰撞）
    
    if verbose:
        print(f"\n[执行] 目标位置: [{pos_mm[0]:.2f}, {pos_mm[1]:.2f}, {pos_mm[2]:.2f}] mm")
        print(f"[执行] 目标姿态: rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°")
        print(f"[执行] 移动速度: {move_speed}")
    
    adjusted_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]

    # 移动到抓取位置
    dobot.move_to_pose(*adjusted_pos, rx, ry, rz, speed=move_speed)
    
    if dobot.check_pose(*adjusted_pos):
        if verbose:
            print("[执行] 到达指定抓取物体位置")
        if enable_gripper:
            gripper.control(gripper_open_pos, gripper_force, gripper_speed)
    
    # 最终位置（要不要去掉安全距离）
    final_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]
    dobot.move_to_pose(*final_pos, rx, ry, rz, speed=move_speed)
    
    if dobot.check_pose(*final_pos):
        if verbose:
            pass
            # print("[执行] 再次确认到达抓取位置，执行抓取")
        if enable_gripper:
            gripper.control(gripper_close_pos, gripper_force, gripper_speed)
            
            gripper_target_close_pos = 90
            # 等待夹爪到达目标位置
            timeout, interval = 5.0, 0.1
            elapsed = 0
            while elapsed < timeout:
                current = gripper.read_current_position() #读取本身费时，容易让机械臂移动有顿挫感
                if current and abs(current[0] - gripper_target_close_pos ) < 5:
                    break
                time.sleep(interval)
                elapsed += interval
        if verbose:
            pass
            print("[完成] 抓取操作完成!")
        return True, T_base_ee_ideal
    else:
        if verbose:
            pass
            # print("[失败] 未能到达最终抓取位置")
        return False, T_base_ee_ideal




#------ move_to_pose 换成 ServoP  （伺服模式） ------
def calculate_grasppose_from_objectpose_servop(
    center_pose_array,
    dobot,
    gripper,
    T_ee_cam,
    z_xoy_angle,
    vertical_euler,
    grasp_tilt_angle,
    angle_threshold,
    T_tcp_ee_z,
    T_safe_distance,
    z_safe_distance,
    gripper_close_pos,
    enable_gripper=True,
    verbose=True
):
    """
    从物体位姿计算并执行抓取动作
    
    Args:
        center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        dobot: Dobot机械臂对象
        gripper: 夹爪对象
        T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象)
        z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)
        vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)，默认[-180, 0, -90]
        grasp_tilt_angle: 倾斜抓取角度 (度)，叠加在vertical_euler[0]上, 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度.
        angle_threshold: z轴对齐的角度阈值 (度)
        T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米)
        T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)
        z_safe_distance: 最终移动时z方向的额外安全距离,也是为了抓取物体靠上的部分。 (毫米)
        gripper_close_pos: 夹爪闭合位置 (0-1000)，默认80
        enable_gripper: 是否执行夹爪抓取动作，默认True
        verbose: 是否打印详细信息
    
    Returns:
        success: 是否成功执行抓取
        T_base_ee_ideal: 计算得到的理想末端执行器位姿 (SE3对象)
    """
    
    # 硬编码的控制参数
    move_speed = 5
    gripper_open_pos = 1000
    gripper_force = 10
    gripper_speed = 20
    
    if vertical_euler is None:
        vertical_euler = [-180, 0, -90]
    
    if verbose:
        print("开始执行本次抓取或者移动操作...")
        # print("="*60)
    
    # # ------check part : 打印物体在相机坐标系中的位姿------
    # if verbose:
    #     print("\n[检查] 物体在相机坐标系中的位姿:")
    #     print_pose_info(center_pose_array, "物体位姿 (相机坐标系)")
    
    # ------计算在机器人基系中的object pose------
    T_cam_object = SE3(center_pose_array, check=False)
    # pose_now = dobot.get_pose()  # 获取当前末端执行器位姿
    pose_now = [470, -20, 430, 195, 0, -90]
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
    
    # if verbose:
    #     print(f"\n[信息] 当前机器人位姿: {pose_now}")
    
    # 从当前机器人位姿构造变换矩阵 T_base_ee
    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
        np.array([x_e, y_e, z_e]) / 1000.0,  # 毫米转米
        check=False
    )
    
    # 坐标变换链: T_base_cam = T_base_ee * T_ee_cam
    T_base_cam = T_base_ee * T_ee_cam
    # T_base_obj = T_base_cam * T_cam_obj（物体在机器人基坐标系中的位姿）
    T_base_obj = T_base_cam * T_cam_object
    
    # ------check part : 检查在机器人基系中，object pose的z轴方向------
    # if verbose:
    #     print("\n[检查] 物体在机器人基坐标系中的位姿:")
    #     print_pose_info(T_base_obj, "物体位姿 (机器人基坐标系)")
    
    # ------object pose 调整------
    T_base_obj_array = np.array(T_base_obj, dtype=float)
    
    # 1. 将object pose的z轴调整为垂直桌面朝上
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:3, 2]  # 提取当前z轴方向
    target_z_axis = np.array([0, 0, 1])  # 目标z轴方向（垂直向上）
    # 计算当前z轴与目标z轴的夹角
    z_angle_error = np.degrees(np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)))
    
    if verbose:
        pass
        # print(f"\n[姿态调整] 当前z轴与垂直方向的偏差: {z_angle_error:.2f}°")
    
    if z_angle_error > angle_threshold:
        if verbose:
            pass
            # print("z轴偏差较大，进行对齐...")
        
        # 计算旋转轴（两向量叉乘）
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:  # 两轴几乎平行
            rotation_matrix_new = current_rotation_matrix
        else:
            rotation_axis = rotation_axis / rotation_axis_norm  # 单位化旋转轴
            rotation_angle = np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
            # 构造反对称矩阵K（用于Rodrigues旋转公式）
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            # Rodrigues旋转公式: R = I + sin(θ)K + (1-cos(θ))K²
            R_z_align = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            rotation_matrix_new = np.dot(R_z_align, current_rotation_matrix)
        
        T_base_obj_aligned = np.eye(4)
        T_base_obj_aligned[:3, :3] = rotation_matrix_new
        T_base_obj_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_aligned, check=False)
    else:
        if verbose:
            pass
            # print("z轴偏差在可接受范围内，使用原始姿态")
        T_base_obj_final = T_base_obj
    
    # 2. 将object pose的x,y轴对齐到机器人基坐标系的x,y轴
    rotation_matrix_after_z = np.array(T_base_obj_final.R)
    current_x_axis = rotation_matrix_after_z[:3, 0]  # 提取当前x轴方向
    # 将x轴投影到水平面（xy平面）
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0])
    x_projected_norm = np.linalg.norm(x_projected)
    
    if x_projected_norm > 1e-6:
        x_projected = x_projected / x_projected_norm  # 单位化投影向量
        # 计算投影与基坐标系x轴的夹角
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        # 构造绕z轴旋转矩阵（消除该夹角）
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
        if verbose:
            pass
            # print("x,y轴对齐完成")
    else:
        if verbose:
            pass
            # print("x轴在水平面的投影太小，跳过x,y轴对齐")
    
    # 3. 将object pose绕z轴旋转指定角度
    # if verbose:
    #     print(f"\n[姿态调整] 将object pose绕z轴旋转{z_xoy_angle}度...")
    
    T_base_obj_array = T_base_obj_final.A
    current_rotation = T_base_obj_array[:3, :3]
    current_translation = T_base_obj_array[:3, 3]
    
    # 构造绕z轴旋转的旋转矩阵
    theta = np.radians(z_xoy_angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    new_rotation = np.dot(R_z, current_rotation)  # 左乘以在基坐标系中旋转
    T_base_obj_rotated = np.eye(4)
    T_base_obj_rotated[:3, :3] = new_rotation
    T_base_obj_rotated[:3, 3] = current_translation
    T_base_obj_final = SE3(T_base_obj_rotated, check=False)
    
    # ------check part : 检查调整后的物体姿态------
    # if verbose:
    #     print("\n[检查] 调整后的最终物体姿态:")
    #     print_pose_info(T_base_obj_final, "调整后物体位姿 (机器人基坐标系)")
    
    # ------调整抓取姿态------
    # 在垂直抓取基础上叠加倾斜角度
    tilted_euler = [vertical_euler[0] + grasp_tilt_angle, vertical_euler[1], vertical_euler[2]]
    
    # if verbose:
    #     print(f"\n[抓取姿态] 垂直抓取姿态: {vertical_euler}")
    #     print(f"[抓取姿态] 倾斜角度: {grasp_tilt_angle}°")
    #     print(f"[抓取姿态] 最终抓取姿态: {tilted_euler}")
    
    # 从欧拉角构造抓取姿态（相对于物体坐标系）
    R_target_xyz = R.from_euler('xyz', tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],  # 抓取点在物体中心
        check=False
    )
    
    # ------check part : 检查相对抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 相对抓取姿态 (物体坐标系):")
        print_pose_info(T_object_grasp_ideal, "T_object_grasp_ideal")
    
    # ------计算在机器人基系中，夹爪grasp即tcp的抓取姿态------
    # 坐标变换链: T_base_grasp = T_base_obj * T_obj_grasp
    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    
    # ------check part : 检查最终抓取姿态------
    if verbose:
        pass
        # print("\n[检查] 最终抓取姿态 (机器人基坐标系):")
        print_pose_info(T_base_grasp_ideal, "T_base_grasp_ideal")
    
    # ------计算在机器人基系中，末端执行器ee的抓取姿态------
    # TCP到末端执行器的偏移（z方向）
    T_tcp_ee = SE3(0, 0, T_tcp_ee_z)
    T_safe_distance = SE3(0, 0, T_safe_distance)  # 额外安全距离
    # 变换链: T_base_ee = T_base_grasp * T_grasp_tcp * T_tcp_ee * T_safe
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe_distance
    
    # ------执行抓取动作------
    pos_mm = T_base_ee_ideal.t * 1000  # 转换为毫米
    # 提取ZYX欧拉角（机械臂使用的旋转顺序）
    rx, ry, rz = T_base_ee_ideal.rpy(unit='deg', order='zyx')
    rz = normalize_angle(rz)  # 规范化到[-180, 180]度
    
    pos_mm[2] += z_safe_distance  # 添加z方向额外安全距离（避免碰撞）
    
    if verbose:
        print(f"\n[执行] 目标位置: [{pos_mm[0]:.2f}, {pos_mm[1]:.2f}, {pos_mm[2]:.2f}] mm")
        print(f"[执行] 目标姿态: rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°")
        print(f"[执行] 移动速度: {move_speed}")
    
    adjusted_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]

    if enable_gripper:
        print("初始抓取位姿: ", *adjusted_pos, rx, ry, rz)

    # ServoP 控制参数设置
    t_control = 0.05      # 控制周期  (比如：0.02s -> 50Hz）
    gain = 200       # 比例增益：数值越小，移动越平滑但也越慢
    aheadtime = 100.0     # 数值越大，移动越平滑
    tolerance = 7      # 到达判定阈值 (mm)
    max_duration = 10.0   # 最大运动时间 (秒)

    

    rospy.init_node('move_to_pose_servo', anonymous=True)
    

    # print(123) #? 会不会被block
    # 移动到抓取位置（使用 move_to_pose_servo）
    if dobot.move_to_pose_servo(*adjusted_pos, rx, ry, rz, 
                                 t=t_control, 
                                 aheadtime=aheadtime, 
                                 gain=gain,
                                 tolerance=tolerance,
                                 max_duration=max_duration,
                                 verbose=verbose):
        # print(456)
        if verbose:
            print("[执行] 到达指定抓取物体位置")
    # print(789)
    

    # 最终位置（要不要去掉安全距离？？？）
    final_pos = [pos_mm[0], pos_mm[1], pos_mm[2]]

    

    # 移动到最终抓取位置（使用 move_to_pose_servo）
    if dobot.move_to_pose_servo(*final_pos, rx, ry, rz, 
                                 t=t_control, 
                                 aheadtime=aheadtime, 
                                 gain=gain,
                                 tolerance=tolerance,
                                 max_duration=max_duration,
                                 verbose=verbose):
        if verbose:
            pass
            # print("[执行] 到达抓取位置，执行抓取")
        
        if enable_gripper:
            gripper.control(gripper_close_pos, gripper_force, gripper_speed)
            
            gripper_target_close_pos = 90
            # 等待夹爪到达目标位置（同时持续发送ServoP保持机械臂位置）
            timeout, interval = 5.0, t_control  # 使用与ServoP相同的频率
            elapsed = 0
            while elapsed < timeout:
                # 持续发送ServoP保持位置，避免飘移
                dobot.ServoP(*final_pos, rx, ry, rz, t=t_control, gain=gain, aheadtime=aheadtime)
                
                current = gripper.read_current_position()
                if current and abs(current[0] - gripper_target_close_pos) < 5:
                    break
                time.sleep(interval)
                elapsed += interval
        
        if verbose:
            pass
            # print("[完成] 抓取操作完成!")
        return True, T_base_ee_ideal
    else:
        if verbose:
            print("[失败] 未能到达最终抓取位置")
        return False, T_base_ee_ideal





#------ 通过视触觉传感器计算玻璃棒倾斜角度------
def detect_dent_orientation(img, save_dir=None):
    # 设置matplotlib支持中文显示（抑制警告）
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    #将图像经过高斯模糊处理
    img = cv2.GaussianBlur(img, (3, 3), 0) #适度模糊（模糊可以减少噪声造成的误判），保留边缘细节
    #检测图像中物体的边缘（轮廓线），将图像转换为只包含边缘信息的二值图像
    edges = cv2.Canny(img, 52, 160)  #低阈值threshold1，高阈值threshold2（降低高阈值提高检测率）
    
    # 保存边缘检测结果到指定文件夹
    if save_dir:
        import os
        edges_path = os.path.join(save_dir, 'edges_detected.png')
        cv2.imwrite(edges_path, edges)
    else:
        cv2.imwrite('edges_detected.png', edges)  #默认保存到当前目录
    # cv2.imshow('edges', edges) #只有边缘线条的黑白图像
    #cv2.waitKey(1)
    
    # 霍夫直线检测
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold= 62)
    print(lines)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            
            # 转换为0-180度范围
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
            
            # 可视化直线
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) #将检测到的line画在原img上
    
    # 保存画了直线的图像
    if save_dir:
        import os
        lines_path = os.path.join(save_dir, 'lines_detected.png')
        cv2.imwrite(lines_path, img)
    
    # 分析角度分布
    if angles:
        avg_angle = np.mean(angles)
        print(f"✅ 平均朝向角度: {avg_angle:.2f}度 (检测到 {len(angles)} 条直线)")
        
        # # 显示结果
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Lines')
        
        # plt.subplot(1, 2, 2)
        # plt.hist(angles, bins=20, alpha=0.7)
        # plt.xlabel('Angle (degrees)')
        # plt.ylabel('Frequency')
        # plt.title('Angle Distribution')
        # plt.show()
    else:
        avg_angle = 0.0  # 未检测到直线时，平均角度设为0
        print("❌ 未检测到直线")
    
    return angles, avg_angle


def adjust_to_vertical_and_lift(
    dobot,
    avg_angle,
    grasp_tilt_angle,
    x_adjustment=115,
    z_adjustment=180,
    adjust_speed=12,
    wait_time=3,
    verbose=True
):
    """
    调整物体姿态至垂直并抬升到安全高度
    
    Args:
        dobot: Dobot机械臂对象
        avg_angle: 检测到的物体当前角度（度）
        grasp_tilt_angle: 抓取时的倾斜角度（度）
        x_adjustment: 抬升后x方向调整量 (mm)，默认115
        z_adjustment: 抬升后z方向调整量 (mm)，默认180
        adjust_speed: 移动速度，默认12
        wait_time: 等待时间（秒），如果为None则根据速度自动计算
        verbose: 是否打印详细信息
    
    Returns:
        dict: {
            'success': bool,           # 是否成功
            'initial_pose': list,      # 调整前的位姿
            'target_pose': list,       # 目标位姿
            'final_pose': list,        # 调整后的实际位姿
            'delta_angle': float,      # 角度调整量
            'x_adjustment': float,     # x方向调整量
            'z_adjustment': float      # z方向调整量
        }
    """
    if verbose:
        print("开始调整物体姿态至垂直桌面向下")
     
    
    # 获取当前位姿
    pose_now = dobot.get_pose()
    if verbose:
        print(f"当前位姿: x={pose_now[0]:.2f}, y={pose_now[1]:.2f}, z={pose_now[2]:.2f}")
        print(f"当前姿态: rx={pose_now[3]:.2f}°, ry={pose_now[4]:.2f}°, rz={pose_now[5]:.2f}°")
    
    # 计算需要调整的角度
    delta_ee = avg_angle - grasp_tilt_angle
    if verbose:
        print(f"\n角度调整计算:")
        print(f"  检测角度 (avg_angle): {avg_angle:.2f}°")
        print(f"  抓取倾角 (grasp_tilt_angle): {grasp_tilt_angle:.2f}°")
        print(f"  需要调整 (delta_ee): {delta_ee:.2f}°")
    
    # 构造目标位姿
    # 需要让tcp朝外旋转；grasp_tilt_angle为正值时，tcp会朝外旋转
    pose_target = [
        pose_now[0] + x_adjustment,
        pose_now[1],
        pose_now[2] + z_adjustment,
        pose_now[3] + delta_ee,
        pose_now[4],
        pose_now[5]
    ]
    
    if verbose:
        print(f"\n目标位姿:")
        print(f"  位置调整: x+{x_adjustment}, z+{z_adjustment}")
        print(f"  目标位置: x={pose_target[0]:.2f}, y={pose_target[1]:.2f}, z={pose_target[2]:.2f}")
        print(f"  目标姿态: rx={pose_target[3]:.2f}°, ry={pose_target[4]:.2f}°, rz={pose_target[5]:.2f}°")
    
    # 执行移动
    dobot.move_to_pose(
        pose_target[0], pose_target[1], pose_target[2],
        pose_target[3], pose_target[4], pose_target[5],
        speed=adjust_speed, acceleration=1
    )
    
    # 等待移动完成
    if wait_time is None:
        # 根据速度自动计算等待时间（速度越慢，等待时间越长）
        wait_time = 1.0 / adjust_speed if adjust_speed > 0 else 0.1
    
    wait_rate = rospy.Rate(1.0 / wait_time)
    wait_rate.sleep()
    
    # 验证是否到达目标位置
    pose_after_adjust = dobot.get_pose()
    angle_error = abs(pose_after_adjust[3] - pose_target[3])
    
    if verbose:
        print(f"\n姿态调整完成:")
        print(f"  实际Rx: {pose_after_adjust[3]:.2f}° (目标: {pose_target[3]:.2f}°)")
        print(f"  角度误差: {angle_error:.2f}°")
        if angle_error < 2.0:
            print("  ✅ 姿态调整精度良好")
        else:
            print("  ⚠️  姿态调整存在偏差")
    
    success = angle_error < 5.0  # 5度以内认为成功
    
    return {
        'success': success,
        'initial_pose': pose_now,
        'target_pose': pose_target,
        'final_pose': pose_after_adjust,
        'delta_angle': delta_ee,
        'x_adjustment': x_adjustment,
        'z_adjustment': z_adjustment,
        'angle_error': angle_error
    }


def descend_with_force_feedback(
    dobot,
    move_step=1,
    max_steps=700,
    force_threshold=1.0,
    sample_interval=0.03,
    max_force_samples=30,
    consecutive_hits_required=2,
    speed=5,
    verbose=True
):
    """
    垂直下降并通过力传感器检测表面接触
    
    Args:
        dobot: Dobot机械臂对象
        move_step: 每步下降距离 (mm)，默认1
        max_steps: 最大下降步数，默认700
        force_threshold: 力阈值 (N)，默认1.0
        sample_interval: 力采样间隔 (秒)，默认0.03
        max_force_samples: 每步最大采样次数，默认30
        consecutive_hits_required: 连续检测到力的次数要求，默认2
        speed: 移动速度，默认5
        verbose: 是否打印详细信息
    
    Returns:
        dict: {
            'contact_detected': bool,      # 是否检测到接触
            'contact_force': float,        # 接触时的力值 (N)
            'descent_distance': float,     # 总下降距离 (mm)
            'steps_taken': int,            # 实际步数
            'initial_pose': list,          # 初始位姿
            'final_pose': list,            # 最终位姿
            'force_history': list          # 力值历史记录（可选）
        }
    """
    if verbose:
        print("\n" + "="*60)
        print("开始垂直下降并监测力反馈...")
        print("="*60)
        print(f"参数配置:")
        print(f"  下降步长: {move_step} mm")
        print(f"  最大步数: {max_steps}")
        print(f"  力阈值: {force_threshold} N")
        print(f"  采样间隔: {sample_interval} s")
        print(f"  连续检测要求: {consecutive_hits_required} 次")
    
    # 记录初始位姿
    pose_initial = dobot.get_pose()
    pose_current = pose_initial.copy()
    
    if verbose:
        print(f"\n初始位置: x={pose_initial[0]:.2f}, y={pose_initial[1]:.2f}, z={pose_initial[2]:.2f}")
    
    # 初始化检测变量
    contact_detected = False
    contact_force = 0.0
    steps_taken = 0
    force_history = []
    
    # 开始逐步下降
    for step in range(max_steps):
        # 控制循环频率
        wait = rospy.Rate(33)
        wait.sleep()
        
        # 下降一步
        pose_current[2] -= move_step
        dobot.move_to_pose(
            pose_current[0], pose_current[1], pose_current[2],
            pose_current[3], pose_current[4], pose_current[5],
            speed=speed, acceleration=1
        )
        
        steps_taken = step + 1
        
        # 采样力传感器数据
        consecutive_hits = 0
        for sample_idx in range(max_force_samples):
            short_wait = rospy.Rate(1 / sample_interval)
            short_wait.sleep()
            
            force_values = dobot.get_force()
            if not force_values:
                continue
            
            # 计算最大力分量
            max_force_component = max(abs(value) for value in force_values)
            force_history.append(max_force_component)
            
            if verbose and sample_idx == 0:  # 只打印每步的第一个采样
                print(f"  步骤 {steps_taken}/{max_steps}: z={pose_current[2]:.2f}mm, F_max={max_force_component:.2f}N", end="")
            
            # 检查是否超过阈值
            if max_force_component >= force_threshold:
                consecutive_hits += 1
                contact_force = max_force_component
                
                if consecutive_hits >= consecutive_hits_required:
                    contact_detected = True
                    if verbose:
                        print(f" ✅ 检测到接触！")
                    break
            else:
                consecutive_hits = 0
        
        if verbose and not contact_detected:
            print()  # 换行
        
        # 如果检测到接触，退出循环
        if contact_detected:
            if verbose:
                print(f"\n接触检测成功:")
                print(f"  下降步数: {steps_taken}")
                print(f"  下降距离: {steps_taken * move_step} mm")
                print(f"  接触力: {contact_force:.2f} N")
                print(f"  最终z坐标: {pose_current[2]:.2f} mm")
            break
    else:
        # 达到最大步数仍未检测到接触
        if verbose:
            print(f"\n⚠️ 达到最大移动距离 ({max_steps * move_step} mm)，未检测到明显接触")
    
    # 获取最终位姿
    pose_final = dobot.get_pose()
    descent_distance = steps_taken * move_step
    
    if verbose:
        print("="*60)
    
    return {
        'contact_detected': contact_detected,
        'contact_force': contact_force,
        'descent_distance': descent_distance,
        'steps_taken': steps_taken,
        'initial_pose': pose_initial,
        'final_pose': pose_final,
        'force_history': force_history if len(force_history) < 1000 else []  # 避免返回过大数据
    }
