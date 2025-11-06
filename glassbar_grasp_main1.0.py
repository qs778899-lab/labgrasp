#!/usr/bin/env python3
import sys
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD   
from create_camera import CreateRealsense
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
import json
import threading
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



# ---------- 手眼标定 ----------
def load_hand_eye_calibration(json_path="hand_eye_calibration.json"):
    """从JSON文件加载手眼标定矩阵"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    calibration = data['T_ee_cam']
    rotation_matrix = np.array(calibration['rotation_matrix'])
    translation_vector = calibration['translation_vector']
    return SE3.Rt(rotation_matrix, translation_vector, check=False)

# 从相机坐标系到末端执行器坐标系的变换矩阵
T_ee_cam = load_hand_eye_calibration()

# ---------- 机械臂 ----------
def init_robot():
    dobot = SimpleApi("192.168.5.1", 29999)
    dobot.clear_error()
    dobot.enable_robot()
    dobot.stop()
    # 启动力传感器
    dobot.enable_ft_sensor(1)
    time.sleep(1)
    # 力传感器置零(以当前受力状态为基准)
    dobot.six_force_home()
    time.sleep(1)
    # 力监控线程
    # force_monitor = ForceMonitor(dobot)
    # force_monitor.start_monitoring()
    # error_monitor = ErrorMonitor(dobot)
    # error_monitor.start_monitoring()
    gripper = DobotGripper(dobot)
    gripper.connect(init=True)
    return dobot, gripper


# ---------- ROS节点 ----------
class ROSSubscriberTest:
    def __init__(self):
        """初始化ROS节点和订阅者"""
        rospy.init_node('ros_subscriber_test', anonymous=True)
        
        # 初始化cv_bridge用于图像转换
        self.bridge = CvBridge()
        
        #? 线程安全的数据存储
        self.latest_tracking_data = {
            'angle_z_deg': 0.0,
            'b': 0.0,
            'x': 0.0,
            'y': 0.0,
            'timestamp': 0.0,
            'valid': False
        }
        self.data_lock = threading.Lock()
        
        # 订阅tracking_data topic
        self.tracking_sub = rospy.Subscriber(
            'tracking_data', 
            Float64MultiArray, 
            self.tracking_callback
        )
        
        # 订阅object_orientation topic  
        self.image_sub = rospy.Subscriber(
            'image_object_orientation',
            Image,
            self.image_callback
        )
        
        print("ROS订阅者已启动，等待数据...")
        
    def tracking_callback(self, msg):
        """处理tracking_data消息"""
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            b = msg.data[1] 
            x = msg.data[2]
            y = msg.data[3]
            
            #? 线程安全地更新数据
            with self.data_lock:
                self.latest_tracking_data.update({
                    'angle_z_deg': angle_z_deg,
                    'b': b,
                    'x': x,
                    'y': y,
                    'timestamp': time.time(),
                    'valid': True
                })
            
            # print(f"📊 跟踪数据: 角度={angle_z_deg:.2f}°, 截距={b:.6f}, 位置=({x:.6f}, {y:.6f})")
        else:
            pass
            # print(f"⚠️  跟踪数据格式错误，期望4个值，实际收到{len(msg.data)}个值")
    
    def image_callback(self, msg):
        """处理object_orientation图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示图像
            cv2.imshow("Object Orientation", cv_image)
            cv2.waitKey(1)  # 非阻塞等待，允许其他处理
            
            # 打印图像信息
            height, width = cv_image.shape[:2]
            print(f"🖼️  接收到图像: {width}x{height} pixels")
            
        except Exception as e:
            print(f"❌ 图像处理错误: {e}")
    
    #?
    def get_latest_tracking_data(self):
        """获取最新的跟踪数据（线程安全）"""
        with self.data_lock:
            return self.latest_tracking_data.copy()
    
    def run(self):
        """运行订阅者"""
        try:
            # 非阻塞保活循环：等待ROS事件，但不主动退出
            while not rospy.is_shutdown():
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n ros中断,正在退出...")


if __name__ == "__main__":
    camera = CreateRealsense("231522072272") #已初始化相机。 相机分辨率是多少？
    mesh_file = "mesh/obj_C_07_02_G.obj"
    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_file)
    mesh.vertices /= 1000 #! 单位转换除以1000
    #! 玻璃棒尺寸更小，但如果mesh已经对了的话就不用缩小。玻璃棒尺寸文件: obj_C_07_02_G.obj
    # mesh.vertices /= 3
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # 初始化机械臂
    dobot, gripper = init_robot()

    #? 初始化ROS订阅者（在后台daemon线程运行，不会阻塞main程序）
    try:
        ros_subscriber = ROSSubscriberTest()
        ros_thread = threading.Thread(target=ros_subscriber.run, daemon=True)
        ros_thread.start()
        print("✅ ROS订阅者已在后台启动（非阻塞模式）")
        time.sleep(1)  # 短暂等待ROS节点启动
    except Exception as e:
        print(f"⚠️  ROS订阅者启动失败: {e}")
        # 创建一个空的占位对象，防止后续代码出错
        class DummySubscriber:
            def get_latest_tracking_data(self):
                return {'valid': False, 'angle_z_deg': 0.0, 'b': 0.0, 'x': 0.0, 'y': 0.0, 'timestamp': 0.0}
        ros_subscriber = DummySubscriber()

    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() 
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    # 获取相机内参
    cam_k = np.loadtxt(f'cam_K.txt').reshape(3,3)

    
    try:
        frame_count = 0
        last_valid_pose = None  # 保存上一次有效的pose
        last_valid_angle = None  # 保存上一次有效的ROS角度
        last_seen_ts = None  # 上一次使用的ROS时间戳
        
        while True:
            # 获取当前帧
            # color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            # depth = camera.get_frames()['depth']/1000
            # ir1 = camera.get_frames()['ir1']
            # ir2 = camera.get_frames()['ir2']
            # 获取当前帧（一次调用，复用返回的所有通道）
            frames = camera.get_frames()
            if frames is None:
                continue
            color = frames['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            depth = frames['depth']/1000
            ir1 = frames['ir1']
            ir2 = frames['ir2']
            
            # cv2.imwrite("ir1.png", ir1)
            # cv2.imwrite("ir2.png", ir2)
            
            
            # 每隔70帧进行一次FoundationPose检测
            if frame_count % 70 == 0:
                #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
                mask = get_mask_from_GD(color, "stirring rod")
                # mask = get_mask_from_GD(color, "Plastic dropper")
            
                cv2.imshow("mask", mask)
                pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
                print(f"第{frame_count}帧检测完成，pose: {pose}")
                center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
                vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('1', vis[...,::-1])
                # cv2.waitKey(0) #waitKey(0) 是一种阻塞
                # input("break001") #input也是一种阻塞
                # print("break001")
                last_valid_pose = center_pose  # 保存这次检测的结果
            else:
                # 使用上一次检测的结果
                center_pose = last_valid_pose
                print(f"第{frame_count}帧使用上次检测结果")
            

            print("center_pose_object: ", center_pose) 
            frame_count += 1

            key = cv2.waitKey(1)
            # if key == ord('q'):  # 按q退出
            #     break
            # elif key == ord('a'):  # 按a执行抓取

            #? 获取ROS跟踪数据（非阻塞，不会停止main程序）
            tracking_data = ros_subscriber.get_latest_tracking_data()
            
            has_new_msg = tracking_data['valid'] and (
                last_seen_ts is None or tracking_data['timestamp'] > last_seen_ts
            )
            
            if has_new_msg:
                # 收到新ROS数据，更新并使用最新角度
                angle_z_deg = tracking_data['angle_z_deg']
                last_valid_angle = angle_z_deg
                last_seen_ts = tracking_data['timestamp']
                print(f"🔄 使用ROS跟踪角度: {angle_z_deg:.2f}° ")
            else:
                # 没有新ROS数据
                if last_valid_angle is not None:
                    angle_z_deg = last_valid_angle
                    print(f"使用上次ROS角度: {angle_z_deg:.2f}° (当前无新数据)")
                else:
                    angle_z_deg = -45  # 朝里
                    print("从未接收到ROS数据，使用默认角度: -45°")

            
            # 将center_pose转换为numpy数组
            center_pose_array = np.array(center_pose, dtype=float)
            
            # ------使用封装函数执行抓取------
            # 配置抓取参数
            z_xoy_angle = 0 # 物体绕z轴旋转角度
            vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
            grasp_tilt_angle = 30  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度： 加了30度会朝外
            z_safe_distance= 15  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
            
            # 调用封装函数执行抓取
            # success, T_base_ee_ideal = execute_grasp_from_object_pose(
            #     center_pose_array=center_pose_array,
            #     dobot=dobot,
            #     gripper=gripper,
            #     T_ee_cam=T_ee_cam,
            #     z_xoy_angle=z_xoy_angle,
            #     vertical_euler=vertical_euler,
            #     grasp_tilt_angle=grasp_tilt_angle,
            #     angle_threshold=10.0,
            #     T_tcp_ee_z= -0.16, 
            #     T_safe_distance=0.003, #可灵活调整
            #     z_safe_distance=z_safe_distance,
            #     verbose=True
            # )
            

            #调整玻璃棒姿态至垂直向下: 
            #当垂直向下，angle为-90度时
            angle_z_deg = -45 #朝里
            target_angle_z_deg = -90
            delta_angle_z_deg = target_angle_z_deg - angle_z_deg #-90+45=-45
            #需要让tcp朝外旋转
            delta_ee = -delta_angle_z_deg - grasp_tilt_angle

            #!当垂直向下，angle为90度时
            

            pose_now = dobot.get_pose()
            pose_target = [pose_now[0], pose_now[1], pose_now[2], pose_now[3]+delta_ee, pose_now[4], pose_now[5] ]

            # dobot.move_to_pose(pose_target[0], pose_target[1], pose_target[2], pose_target[3], pose_target[4], pose_target[5], speed=7, acceleration=1) 


            # 移动玻璃棒到指定位置

            success = True
            if success:
                print("\n[成功] 抓取操作完成!")
                # input("按Enter继续...") #不适合在循环中使用
            else:
                print("\n[失败] 抓取操作未完成")
                
            # 可选：返回home位置（根据需要取消注释）
            # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)
    

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        cv2.destroyAllWindows()
        # dobot.disable_robot()

    #-------run demo---------

