#!/usr/bin/env python3
"""
ForceField压缩图像接收测试
测试CompressedImage格式的ROS通信
"""

import cv2
import numpy as np
import rospy
import threading
import time
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from typing import Optional

class ForceFieldCompressedReceiver:
    """ForceField压缩图像接收器"""
    
    def __init__(self):
        # ROS初始化
        rospy.init_node('forcefield_compressed_receiver', anonymous=True)
        
        # 三个独立的图像帧
        self.tactile_frame = None
        self.normal_frame = None
        self.shear_frame = None
        self.lock = threading.Lock()
        
        # 状态
        self.is_running = False
        self.forcefield_available = False
        
        # ROS订阅者 - 三个压缩图像话题
        self.tactile_sub = rospy.Subscriber(
            '/forcefield/tactile_image/compressed', 
            CompressedImage, 
            self.tactile_callback
        )
        self.normal_sub = rospy.Subscriber(
            '/forcefield/normal_force/compressed', 
            CompressedImage, 
            self.normal_callback
        )
        self.shear_sub = rospy.Subscriber(
            '/forcefield/shear_force/compressed', 
            CompressedImage, 
            self.shear_callback
        )
        
        # ROS发布者
        self.status_pub = rospy.Publisher(
            '/forcefield/compressed_receiver_status', 
            String, 
            queue_size=1
        )
        
        # 启动状态发布线程
        self.status_thread = threading.Thread(target=self._publish_status, daemon=True)
        self.status_thread.start()
        
        print("✅ ForceField压缩图像接收器初始化完成")
    
    def tactile_callback(self, msg):
        """触觉图像回调函数"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # 转换回RGB格式用于显示
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.tactile_frame = cv_image.copy()
                    self.forcefield_available = True
                    print(f"✅ 接收到触觉图像: {cv_image.shape}")
        except Exception as e:
            rospy.logerr(f"触觉图像解码失败: {e}")
    
    def normal_callback(self, msg):
        """法向力图像回调函数"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                with self.lock:
                    self.normal_frame = cv_image.copy()
                    print(f"✅ 接收到法向力图像: {cv_image.shape}")
        except Exception as e:
            rospy.logerr(f"法向力图像解码失败: {e}")
    
    def shear_callback(self, msg):
        """剪切力图像回调函数"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                with self.lock:
                    self.shear_frame = cv_image.copy()
                    print(f"✅ 接收到剪切力图像: {cv_image.shape}")
        except Exception as e:
            rospy.logerr(f"剪切力图像解码失败: {e}")
    
    def _publish_status(self):
        """发布状态信息"""
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            try:
                status_msg = String()
                if self.forcefield_available:
                    status_msg.data = "compressed_receiver_available"
                else:
                    status_msg.data = "compressed_receiver_unavailable"
                
                self.status_pub.publish(status_msg)
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"状态发布失败: {e}")
                rate.sleep()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧（拼接的三个图像）"""
        with self.lock:
            if self.tactile_frame is not None and self.normal_frame is not None and self.shear_frame is not None:
                # 水平拼接三个图像
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
        """启动接收器"""
        if not self.is_running:
            self.is_running = True
            print("🚀 ForceField压缩图像接收器已启动")
            return True
        return False
    
    def stop(self):
        """停止接收器"""
        self.is_running = False
        rospy.signal_shutdown("接收器停止")

# 全局实例
forcefield_compressed_receiver = ForceFieldCompressedReceiver()

def main():
    """压缩图像接收器主函数"""
    try:
        print("🚀 启动ForceField压缩图像接收器...")
        print("📡 等待ForceField节点发布压缩图像...")
        print("💡 提示：请确保ForceField节点正在运行并发布到以下话题:")
        print("   - /forcefield/tactile_image/compressed (触觉图像)")
        print("   - /forcefield/normal_force/compressed (法向力)")
        print("   - /forcefield/shear_force/compressed (剪切力)")
        
        # 启动接收器
        forcefield_compressed_receiver.start()
        
        # 显示循环
        frame_count = 0
        while not rospy.is_shutdown():
            frame = forcefield_compressed_receiver.get_current_frame()
            if frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧打印一次状态
                    status = forcefield_compressed_receiver.get_status()
                    print(f"📊 状态: {status}")
                
                cv2.imshow('ForceField Compressed Receiver - 三个并排画面', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 用户退出")
                    break
            else:
                # 显示等待信息
                wait_img = np.zeros((480, 1920, 3), dtype=np.uint8)
                cv2.putText(wait_img, 'Waiting for ForceField Compressed Images...', 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(wait_img, 'Topics: tactile_image/compressed, normal_force/compressed, shear_force/compressed', 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(wait_img, 'Press q to quit', 
                           (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('ForceField Compressed Receiver - 三个并排画面', wait_img)
                
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
        forcefield_compressed_receiver.stop()
        cv2.destroyAllWindows()
        print("🧹 资源清理完成")

if __name__ == "__main__":
    main()
