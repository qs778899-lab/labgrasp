#!/usr/bin/env python3
"""
ForceField ROS桥接模块
通过ROS通信获取ForceField三个并排画面
"""

import cv2
import numpy as np
import rospy
import threading
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool, String
from typing import Optional

class ForceFieldROSBridge:
    """ForceField ROS桥接类"""
    
    def __init__(self):
        # ROS初始化
        rospy.init_node('forcefield_ros_bridge', anonymous=True)
        
        # CV桥接器
        self.bridge = CvBridge()
        
        # 三个独立的图像帧
        self.tactile_frame = None
        self.normal_frame = None
        self.shear_frame = None
        self.lock = threading.Lock()
        
        # 状态
        self.is_running = False
        self.forcefield_available = False
        
        # ROS订阅者 - 三个独立话题
        self.tactile_sub = rospy.Subscriber(
            '/forcefield/tactile_image', 
            Image, 
            self.tactile_callback
        )
        self.normal_sub = rospy.Subscriber(
            '/forcefield/normal_force', 
            Image, 
            self.normal_callback
        )
        self.shear_sub = rospy.Subscriber(
            '/forcefield/shear_force', 
            Image, 
            self.shear_callback
        )
        
        # ROS发布者
        self.status_pub = rospy.Publisher(
            '/forcefield/web_status', 
            String, 
            queue_size=1
        )
        
        # # 启动状态发布线程
        # self.status_thread = threading.Thread(target=self._publish_status, daemon=True)
        # self.status_thread.start()
        
        self.normal_frame = None



        print("✅ ForceField ROS桥接器初始化完成")
    
    def tactile_callback(self, msg):
        """触觉图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            with self.lock:
                self.tactile_frame = cv_image.copy()
                self.forcefield_available = True
        except Exception as e:
            rospy.logerr(f"触觉图像转换失败: {e}")
    
    def normal_callback(self, msg):
        """法向力图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.normal_frame = cv_image.copy()
        except Exception as e:
            rospy.logerr(f"法向力图像转换失败: {e}")
    
    def shear_callback(self, msg):
        """剪切力图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.shear_frame = cv_image.copy()
        except Exception as e:
            rospy.logerr(f"剪切力图像转换失败: {e}")
    
    # def _publish_status(self):
    #     """发布状态信息"""
    #     rate = rospy.Rate(1)  # 1Hz
    #     while not rospy.is_shutdown():
    #         try:
    #             status_msg = String()
    #             if self.forcefield_available:
    #                 status_msg.data = "available"
    #             else:
    #                 status_msg.data = "unavailable"
                
    #             self.status_pub.publish(status_msg)
    #             rate.sleep()
                
    #         except Exception as e:
    #             rospy.logerr(f"状态发布失败: {e}")
    #             rate.sleep()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧（拼接的三个图像）"""
        with self.lock:
            if self.tactile_frame is not None and self.normal_frame is not None and self.shear_frame is not None:
                # 水平拼接三个图像
                print("here")
                combined_frame = cv2.hconcat([self.tactile_frame, self.normal_frame, self.shear_frame])
                return combined_frame
            return None
    
    def get_three_frames(self) -> dict:
        """获取三个独立的图像帧"""
        with self.lock:
            return {
                'tactile': self.tactile_frame.copy() if self.tactile_frame is not None else None,
                'normal': self.normal_frame.copy() if self.normal_frame is not None else None,
                'shear': self.shear_frame.copy() if self.shear_frame is not None else None
            }
    
    def get_tactile_frame(self) -> Optional[np.ndarray]:
        """获取触觉图像"""
        with self.lock:
            return self.tactile_frame.copy() if self.tactile_frame is not None else None
    
    def get_normal_frame(self) -> Optional[np.ndarray]:
        """获取法向力图像"""
        with self.lock:
            return self.normal_frame.copy() if self.normal_frame is not None else None
    
    def get_shear_frame(self) -> Optional[np.ndarray]:
        """获取剪切力图像"""
        with self.lock:
            return self.shear_frame.copy() if self.shear_frame is not None else None
    
    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            'ros_connected': not rospy.is_shutdown(),
            'forcefield_available': self.forcefield_available,
            'has_tactile': self.tactile_frame is not None,
            'has_normal': self.normal_frame is not None,
            'has_shear': self.shear_frame is not None,
            'node_name': rospy.get_name()
        }
    
    def start(self) -> bool:
        """启动ROS桥接器"""
        if not self.is_running:
            self.is_running = True
            print("🚀 ForceField ROS桥接器已启动")
            return True
        return False
    
    def stop(self):
        """停止ROS桥接器"""
        self.is_running = False
        rospy.signal_shutdown("Web应用停止")

# 全局实例
forcefield_ros_bridge = ForceFieldROSBridge()

def main():
    """ROS桥接器主函数"""
    try:
        print("🚀 启动ForceField ROS桥接器...")
        print("📡 等待ForceField节点发布图像...")
        print("💡 提示：请确保ForceField节点正在运行并发布到以下话题:")
        print("   - /forcefield/tactile_image (触觉图像)")
        print("   - /forcefield/normal_force (法向力)")
        print("   - /forcefield/shear_force (剪切力)")
        
        # 启动桥接器
        forcefield_ros_bridge.start()
        time.sleep(3)
        
        # 简单的显示循环（用于测试）
        while not rospy.is_shutdown():
            frame = forcefield_ros_bridge.get_current_frame()
            if frame is not None:
                print(frame.shape)
                cv2.imshow('ForceField ROS Bridge',frame )
                
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 用户退出")
                break

            
            rospy.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        forcefield_ros_bridge.stop()
        cv2.destroyAllWindows()
        print("🧹 资源清理完成")

if __name__ == "__main__":
    main()
