#!/usr/bin/env python3
"""
测试ROS数据接收和main循环的时序问题
"""
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time

class SimpleROSTest:
    def __init__(self):
        rospy.init_node('simple_ros_test', anonymous=True)
        
        self.latest_tracking_data = {
            'angle_z_deg': 0.0,
            'valid': False,
            'timestamp': 0.0
        }
        self.data_lock = threading.Lock()
        
        self.tracking_sub = rospy.Subscriber(
            'tracking_data', 
            Float64MultiArray, 
            self.tracking_callback
        )
        
        print("✅ ROS测试节点已启动")
        
    def tracking_callback(self, msg):
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            
            with self.data_lock:
                self.latest_tracking_data.update({
                    'angle_z_deg': angle_z_deg,
                    'valid': True,
                    'timestamp': time.time()
                })
            
            print(f"📊 [ROS回调线程] 收到数据: angle={angle_z_deg:.2f}°, valid=True")
    
    def get_latest_tracking_data(self):
        with self.data_lock:
            return self.latest_tracking_data.copy()
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n退出ROS节点")

def main():
    print("=" * 50)
    print("ROS时序测试")
    print("=" * 50)
    
    # 启动ROS订阅者
    ros_test = SimpleROSTest()
    ros_thread = threading.Thread(target=ros_test.run, daemon=True)
    ros_thread.start()
    time.sleep(1)
    
    print("\n开始main循环测试...")
    
    for i in range(20):
        print(f"\n--- 循环 {i} ---")
        
        # 模拟耗时操作（类似FoundationPose）
        if i % 5 == 0:
            print(f"[Main线程] 开始耗时操作...")
            time.sleep(3)  # 模拟3秒的处理时间
            print(f"[Main线程] 耗时操作完成")
        
        # 获取ROS数据
        tracking_data = ros_test.get_latest_tracking_data()
        print(f"[Main线程] 读取数据: valid={tracking_data['valid']}, angle={tracking_data['angle_z_deg']:.2f}°")
        
        if tracking_data['valid']:
            print(f"✅ [Main线程] 使用ROS角度: {tracking_data['angle_z_deg']:.2f}°")
        else:
            print(f"⚠️ [Main线程] 使用默认角度")
        
        time.sleep(0.5)  # 模拟其他处理
    
    print("\n测试完成")

if __name__ == "__main__":
    main()


