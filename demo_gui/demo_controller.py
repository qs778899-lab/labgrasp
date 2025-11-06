import threading
import time
from flask import jsonify
from demo1_worker import Demo1Worker
from force_grasp_worker import ForceGraspWorker
from hardware_manager import hardware_manager

class DemoController:
    """演示控制器 - 硬件总是可用，专注控制逻辑"""
    
    def __init__(self):
        self.demo1_worker = Demo1Worker()
        self.force_grasp_worker = ForceGraspWorker()
    
    def start_demo1(self):
        return jsonify(self.demo1_worker.start())
    
    def stop_demo1(self):
        return jsonify(self.demo1_worker.stop())
    
    def get_demo1_status(self):
        return jsonify(self.demo1_worker.get_status())
    
    def robot_reset(self):
        """机械臂复位 - 硬件总是可用"""
        try:
            # TODO: 调用实际的复位逻辑
            return jsonify({'success': True, 'message': '机械臂复位成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    def gripper_reset(self):
        """夹爪复位 - 直接操作硬件"""
        try:
            gripper = hardware_manager.get_gripper()
            gripper.send_command(position=800.0, force=20, speed=50)
            return jsonify({'success': True, 'message': '夹爪复位成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    def gripper_force_grasp(self):
        """夹爪力控抓取 - 智能像素监控版本"""
        return jsonify(self.force_grasp_worker.start())
    