import sys
import os
sys.path.append("/home/yimu/wrc/realman_ws/tasks")
from pixel_monitor import PixelMonitor

class HardwareManager:
    """硬件管理器 - 初始化失败则终止，成功则硬件总是可用"""
    
    def __init__(self):
        # 硬件对象
        self.robo_arm = None
        self.handle = None
        self.arm = None
        self.gripper = None
        self.pixel_monitor = None
        
        # 立即初始化，失败则退出
        self._initialize_or_die()
    
    def _initialize_or_die(self):
        """初始化硬件，失败则终止应用"""
        print("=== 初始化硬件系统 ===")
        
        # 导入硬件模块
        try:
            import rospy
            from utils.control_gripper import GC
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            from toolbox.arm_tb import ArmController
            print("✓ 硬件模块导入成功")
        except ImportError as e:
            print(f"❌ 硬件模块导入失败: {e}")
            print("💀 应用终止")
            sys.exit(1)
        
        # 初始化ROS
        try:
            if not rospy.get_node_uri():
                rospy.init_node('demo_hardware_node', anonymous=True)
            print("✓ ROS节点初始化成功")
        except Exception as e:
            print(f"❌ ROS初始化失败: {e}")
            print("💀 应用终止")
            sys.exit(1)
        
        # 初始化机械臂
        try:
            self.robo_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.handle = self.robo_arm.rm_create_robot_arm("192.168.1.18", 8080)
        
            
            self.arm = ArmController(self.robo_arm, self.handle)
            print("✓ 机械臂连接成功")
        except Exception as e:
            print(f"❌ 机械臂初始化失败: {e}")
            print("💀 应用终止")
            sys.exit(1)
        
        # 初始化夹爪
        try:
            self.gripper = GC(arm=self.robo_arm, handle=self.handle, init=True)
            # 测试夹爪
            self.gripper.send_command(position=800.0, force=20, speed=50)
            print("✓ 夹爪初始化成功")
        except Exception as e:
            print(f"❌ 夹爪初始化失败: {e}")
            print("💀 应用终止")
            sys.exit(1)
        
        # 初始化像素监控器
        try:
            self.pixel_monitor = PixelMonitor(camera_id=10)
            print("✓ 像素监控器初始化成功")
        except Exception as e:
            print(f"❌ 像素监控器初始化失败: {e}")
            print("💀 应用终止")
            sys.exit(1)
        
        print("🎉 所有硬件初始化完成！")
    
    def get_gripper(self):
        """获取夹爪对象 - 总是可用"""
        return self.gripper
    
    def get_pixel_monitor(self):
        """获取像素监控器对象 - 总是可用"""
        return self.pixel_monitor
    
    def get_arm(self):
        """获取机械臂对象 - 总是可用"""
        return self.arm

# 全局硬件管理器实例
print("first")
hardware_manager = HardwareManager() 