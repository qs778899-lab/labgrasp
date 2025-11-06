# coding=utf-8
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import threading

def create_folder_with_date():
    """创建时间戳文件夹"""
    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

class CreateRealsense:
    def __init__(self, device_id):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(device_id)
        
        # 配置流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        
        # 启动管道
        self.profile = self.pipeline.start(self.config)
        # 对齐对象：深度→彩色
        self.align = rs.align(rs.stream.color)
        
        # 配置传感器
        self._setup_sensor()
        
        # 预热相机
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("相机初始化完成")
        
        # 线程控制
        self.running = False
        self.frame_lock = threading.Lock()
        self.frames = {'color': None, 'depth': None, 'ir1': None, 'ir2': None}
        self.save_counter = 1

    def _setup_sensor(self):
        """配置传感器参数"""
        sensor = self.profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.emitter_enabled, 0)
        
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"深度缩放因子: {depth_scale}")

    def get_frames(self):
        """获取所有帧数据"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir1_frame = aligned_frames.get_infrared_frame(1)
        ir2_frame = aligned_frames.get_infrared_frame(2)
        
        if not (color_frame and depth_frame and ir1_frame and ir2_frame):
            return None
            
        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': np.asanyarray(depth_frame.get_data()),
            'ir1': np.asanyarray(ir1_frame.get_data()),
            'ir2': np.asanyarray(ir2_frame.get_data())
        }

    def _update_frames(self):
        """后台更新帧数据"""
        while self.running:
            frame_data = self.get_frames()
            if frame_data:
                with self.frame_lock:
                    self.frames = frame_data

    def _save_frames(self, save_dir):
        """保存当前帧"""
        with self.frame_lock:
            frames = self.frames.copy()
            
        if not all(frames.values()):
            return
            
        base_name = os.path.join(save_dir, f"{self.save_counter:04d}")
        
        # 保存文件
        cv2.imwrite(f"{base_name}_rgb.jpg", frames['color'])
        cv2.imwrite(f"{base_name}_depth.png", frames['depth'])
        cv2.imwrite(f"{base_name}_ir1.png", frames['ir1'])
        cv2.imwrite(f"{base_name}_ir2.png", frames['ir2'])
        
        # 保存深度彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(frames['depth'], alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imwrite(f"{base_name}_depthmap.png", depth_colormap)
        
        print(f"已保存第 {self.save_counter} 组图像")
        self.save_counter += 1

    def show_frame(self):
        """显示实时画面"""
        self.running = True
        save_dir = None
        
        # 启动帧更新线程
        update_thread = threading.Thread(target=self._update_frames, daemon=True)
        update_thread.start()
        
        try:
            while self.running:
                with self.frame_lock:
                    frames = self.frames.copy()
                
                if not all(frames.values()):
                    time.sleep(0.01)
                    continue
                
                # 创建显示图像
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(frames['depth'], alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                display1 = np.hstack((frames['color'], depth_colormap))
                display2 = np.hstack((frames['ir1'], frames['ir2']))
                
                cv2.imshow("RGB & Depth", display1)
                cv2.imshow("IR1 & IR2", display2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty("RGB & Depth", cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('s'):
                    if save_dir is None:
                        save_dir = create_folder_with_date()
                        print(f"保存目录: {save_dir}")
                    self._save_frames(save_dir)
                    
        finally:
            self.running = False
            cv2.destroyAllWindows()

    def release(self):
        """释放资源"""
        self.running = False
        self.pipeline.stop()

def main():
    """显示可用的 RealSense 设备"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print("可用的 RealSense 设备:")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"设备 {i}: {name} - 序列号: {serial}")

if __name__ == "__main__":
    main()
    device_id = input("请输入设备序列号: ")
    
    cam = CreateRealsense(device_id)
    try:
        cam.show_frame()
    finally:
        cam.release()
