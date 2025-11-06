#!/usr/bin/env python3
"""
简单的ROS测试
"""

import rospy
import cv2
import numpy as np

def main():
    print("🚀 简单ROS测试开始...")
    
    try:
        # 初始化ROS节点
        rospy.init_node('simple_test', anonymous=True)
        print("✅ ROS节点初始化成功")
        
        # 检查话题
        topics = rospy.get_published_topics()
        print(f"📡 发现 {len(topics)} 个ROS话题:")
        for topic_name, topic_type in topics[:5]:  # 只显示前5个
            print(f"   - {topic_name}")
        
        # 创建测试图像
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, 'ROS Test OK', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # 显示图像
        print("📺 显示测试图像，按 'q' 退出")
        cv2.imshow('Simple ROS Test', test_img)
        
        # 等待按键
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()

