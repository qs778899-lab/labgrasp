#!/usr/bin/env python3
"""
简单的ROS连接测试
测试ForceField ROS桥接器是否能正常工作
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import time

def test_ros_connection():
    """测试ROS连接"""
    print("🚀 开始ROS连接测试...")
    
    try:
        # 初始化ROS节点
        rospy.init_node('test_ros_connection', anonymous=True)
        print("✅ ROS节点初始化成功")
        
        # 创建CV桥接器
        bridge = CvBridge()
        print("✅ CV桥接器创建成功")
        
        # 测试话题列表
        print("📡 当前ROS话题:")
        topics = rospy.get_published_topics()
        for topic_name, topic_type in topics:
            print(f"   - {topic_name} ({topic_type})")
        
        # 检查ForceField话题是否存在
        forcefield_topics = [
            '/forcefield/tactile_image',
            '/forcefield/normal_force', 
            '/forcefield/shear_force'
        ]
        
        found_topics = []
        for topic in forcefield_topics:
            if any(topic_name == topic for topic_name, _ in topics):
                found_topics.append(topic)
        
        if len(found_topics) == 3:
            print("✅ 找到所有ForceField话题:")
            for topic in found_topics:
                print(f"   - {topic}")
        elif len(found_topics) > 0:
            print(f"⚠️ 找到部分ForceField话题 ({len(found_topics)}/3):")
            for topic in found_topics:
                print(f"   - {topic}")
        else:
            print("⚠️ 未找到ForceField话题")
            print("💡 请确保ForceField发布器正在运行")
        
        return True
        
    except Exception as e:
        print(f"❌ ROS连接测试失败: {e}")
        return False

def test_forcefield_bridge():
    """测试ForceField桥接器"""
    print("\n🔗 测试ForceField桥接器...")
    
    try:
        # 导入桥接器
        from forcefield_ros_bridge import ForceFieldROSBridge
        
        # 创建桥接器实例
        bridge = ForceFieldROSBridge()
        print("✅ ForceField桥接器创建成功")
        
        # 获取状态
        status = bridge.get_status()
        print(f"📊 桥接器状态: {status}")
        
        # 启动桥接器
        bridge.start()
        print("✅ ForceField桥接器启动成功")
        
        # 等待一段时间接收数据
        print("⏳ 等待ForceField数据...")
        for i in range(10):
            frame = bridge.get_current_frame()
            if frame is not None:
                print("✅ 成功接收到ForceField图像数据！")
                print(f"📐 图像尺寸: {frame.shape}")
                return True
            else:
                print(f"⏳ 等待中... ({i+1}/10)")
                time.sleep(1)
        
        print("⚠️ 未接收到ForceField数据")
        return False
        
    except Exception as e:
        print(f"❌ ForceField桥接器测试失败: {e}")
        return False

def test_simple_image_display():
    """测试简单图像显示"""
    print("\n🖼️ 测试图像显示...")
    
    try:
        # 创建一个测试图像
        test_image = np.zeros((480, 1920, 3), dtype=np.uint8)
        
        # 添加一些内容
        cv2.putText(test_image, 'ROS Connection Test', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(test_image, 'ForceField ROS Bridge', (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        cv2.putText(test_image, 'Press q to quit', (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示图像
        print("📺 显示测试图像，按 'q' 键退出")
        cv2.imshow('ROS Connection Test', test_image)
        
        # 等待按键
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("✅ 图像显示测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 图像显示测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 ForceField ROS连接测试")
    print("=" * 50)
    
    # 测试1: ROS连接
    test1_result = test_ros_connection()
    
    # 测试2: ForceField桥接器
    test2_result = test_forcefield_bridge()
    
    # 测试3: 图像显示
    test3_result = test_simple_image_display()
    
    print("\n" + "=" * 50)
    print("📋 测试结果:")
    print(f"   ROS连接: {'✅ 成功' if test1_result else '❌ 失败'}")
    print(f"   ForceField桥接器: {'✅ 成功' if test2_result else '❌ 失败'}")
    print(f"   图像显示: {'✅ 成功' if test3_result else '❌ 失败'}")
    
    if test1_result and test2_result and test3_result:
        print("\n🎉 所有测试通过！ROS通信建立成功！")
        print("\n📋 下一步:")
        print("1. 启动ForceField发布器: cd /home/yimu/wrc/sparsh && conda activate tactile && python forcefield_ros_publisher.py")
        print("2. 启动Web应用: cd /home/yimu/new_work/demo_gui && conda activate py311 && python main.py")
        print("3. 访问: http://localhost:5000/demo1")
    else:
        print("\n❌ 部分测试失败，请检查ROS环境配置")
    
    return test1_result and test2_result and test3_result

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        exit(1)

