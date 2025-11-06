import threading
import time
import rospy
from hardware_manager import hardware_manager

class Demo1Worker:
    """Demo1工作线程 - 硬件总是可用，专注业务逻辑"""
    
    def __init__(self):
        self.state = {
            'running': False,
            'thread': None,
            'stop_flag': False,
            'status': '未启动',
            'message': ''
        }
    
    def start(self):
        """启动Demo1"""
        if self.state['running']:
            return {'success': False, 'message': 'Demo1 已在运行中'}
        
        self.state['running'] = True
        self.state['stop_flag'] = False
        self.state['status'] = '启动中...'
        
        self.state['thread'] = threading.Thread(target=self._run_demo)
        self.state['thread'].daemon = True
        self.state['thread'].start()
        
        return {'success': True, 'message': 'Demo1 启动成功'}
    
    def stop(self):
        """停止Demo1"""
        if not self.state['running']:
            return {'success': False, 'message': 'Demo1 未在运行'}
        
        self.state['stop_flag'] = True
        self.state['status'] = '停止中...'
        
        return {'success': True, 'message': 'Demo1 停止指令已发送'}
    
    def get_status(self):
        """获取状态"""
        return {
            'running': self.state['running'],
            'status': self.state['status'],
            'message': self.state['message']
        }
    
    def _run_demo(self):
        """执行演示 - 硬件总是可用"""
        try:
            self.state['status'] = '运行中'
            self.state['message'] = '开始执行夹薯片演示'
            
            # 直接获取硬件，无需检查
            gripper = hardware_manager.get_gripper()
            pixel_monitor = hardware_manager.get_pixel_monitor()
            
            cycle_count = 0
            while not self.state['stop_flag'] and not rospy.is_shutdown():
                cycle_count += 1
                # 移动到初始位置并重置基准图像
                gripper.send_command(position=500.0, force=20, speed=30)
                rospy.sleep(1.0)
                pixel_monitor.reset_baseline()
                
                # 开始夹取，监控像素变化
                self.state['message'] = f'第 {cycle_count} 次循环 - 开始夹取'
                gripper.send_command(position=100.0, force=20, speed=7)
                
                # 简单的像素变化检测
                for _ in range(1000):  # 最多检测10秒 (0.01 * 1000)
                    if self.state['stop_flag']:
                        break
                    if pixel_monitor.has_significant_change():
                        self.state['message'] = f'第 {cycle_count} 次循环 - 检测到变化，暂停'
                        gripper.pause_movement(force=20, speed=20)
                        break
                    rospy.sleep(0.01)
                
                # 复位
                gripper.send_command(position=800.0, force=20, speed=50)
                rospy.sleep(2.0)
                
                if cycle_count >= 50:
                    break
                    
        except Exception as e:
            self.state['status'] = '运行错误'
            self.state['message'] = f'演示执行出错: {str(e)}'
        finally:
            self.state['running'] = False
            self.state['thread'] = None
            if self.state['status'] != '运行错误':
                self.state['status'] = '已停止'
                self.state['message'] = 'Demo1 执行完成或被停止' 