from utils.camera_reader import CameraReader

class PixelMonitor:
    """像素变化监控器"""
    
    def __init__(self, camera_id=10):
        self.camera_reader = CameraReader(camera_id=camera_id)
        print(f"✓ 相机{camera_id}初始化成功")
    
    def reset_baseline(self):
        """重置基准图像"""
        self.camera_reader.reset_init()
    
    def has_significant_change(self):
        """检测是否有显著变化"""
        current_frame = self.camera_reader.get_current_frame()
        if current_frame is None:
            return False
            
        change_result = self.camera_reader.compare_with_init_image(
            current_frame, method='pixel_changes', threshold=30
        )
        
        if 'error' in change_result:
            return False
            
        return (change_result.get('has_changes', False) and 
                change_result.get('change_percentage', 0) > 0.001) 