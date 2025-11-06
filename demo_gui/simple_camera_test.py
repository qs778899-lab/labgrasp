#!/usr/bin/env python3
"""
简单相机测试 - 只显示实时画面
"""

import cv2
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.camera_reader import CameraReader

def main():
    print("🎥 简单相机测试")
    print("按 'q' 键退出")
    
    # 创建相机
    camera = CameraReader(camera_id=10)
    
    # 显示实时画面
    while True:
        frame = camera.get_current_frame()
        if frame is None:
            print("无法获取相机画面")
            break
        
        # 调整窗口大小 - 缩放到屏幕合适大小
        height, width = frame.shape[:2]
        max_width = 800  # 最大宽度
        max_height = 600  # 最大高度
        
        # 计算缩放比例
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow('Camera', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("测试结束")

if __name__ == "__main__":
    main()
