import threading
import time
import rospy
from hardware_manager import hardware_manager

class ForceGraspWorker:
    """一次性力控抓取工作线程"""
    
    def __init__(self):
        self.state = {
            'running': False,
            'thread': None,
            'status': '未启动',
            'message': ''
        }
    
    def start(self):
        """启动一次性抓取"""
        if self.state['running']:
            return {'success': False, 'message': '抓取任务已在运行中'}
        
        self.state['running'] = True
        self.state['status'] = '启动中...'
        
        self.state['thread'] = threading.Thread(target=self._run_grasp)
        self.state['thread'].daemon = True
        self.state['thread'].start()
        
        return {'success': True, 'message': '力控抓取启动成功'}
    
    def get_status(self):
        """获取状态"""
        return {
            'running': self.state['running'],
            'status': self.state['status'],
            'message': self.state['message']
        }
    
    def _run_grasp(self):
        """执行一次性抓取"""
        try:
            self.state['status'] = '运行中'
            self.state['message'] = '开始执行力控抓取'
            
            # 获取硬件
            gripper = hardware_manager.get_gripper()
            pixel_monitor = hardware_manager.get_pixel_monitor()
            
            # 移动到初始位置并重置基准图像
            self.state['message'] = '移动到初始位置'
            gripper.send_command(position=800.0, force=20, speed=30)
            rospy.sleep(1.0)
            pixel_monitor.reset_baseline()
            
            # 开始抓取，监控像素变化
            self.state['message'] = '开始抓取，监控变化...'
            gripper.send_command(position=100.0, force=20, speed=7)
            
            # 像素变化检测
            change_detected = False
            for _ in range(1000):  # 最多检测10秒
                if pixel_monitor.has_significant_change():
                    self.state['message'] = '检测到变化，暂停抓取'
                    gripper.pause_movement(force=20, speed=20)
                    change_detected = True
                    break
                rospy.sleep(0.01)
            
            if not change_detected:
                self.state['message'] = '未检测到变化，抓取完成'
            
            # 保持抓取状态2秒
            rospy.sleep(2.0)
                    
            self.state['status'] = '已完成'
            self.state['message'] = '力控抓取完成'
                    
        except Exception as e:
            self.state['status'] = '运行错误'
            self.state['message'] = f'抓取执行出错: {str(e)}'
        finally:
            self.state['running'] = False
            self.state['thread'] = None 