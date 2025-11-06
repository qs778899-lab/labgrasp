
from flask import Flask, render_template, Response
import sys
import threading
import time
import cv2
import base64

sys.path.append("/home/yimu/wrc/realman_ws/tasks")

from demo_controller import DemoController
from hardware_manager import hardware_manager
from utils.camera_reader import CameraReader
from forcefield_ros_bridge_bridge import forcefield_ros_bridge

app = Flask(__name__)

# 演示控制器 - 硬件已经初始化完成
demo_controller = DemoController()

# 初始化相机
camera_reader = CameraReader(camera_id=10)
print("🚀 Web应用启动完成！")

# === 页面路由 ===
@app.route('/')
def index():
    return render_template('form.html', show_start=True)

@app.route('/operations')
def operations():
    return render_template('form.html', show_start=False)

@app.route('/demo1') ## 当用户点击"Demo1"链接时，浏览器访问 /demo1
def demo1_page():
    return render_template('demo1.html')

@app.route('/demo2')
def demo2():
    return "这是Demo 2的内容"

@app.route('/demo3')
def demo3():
    return "这是Demo 3的内容"

@app.route('/demo4')
def demo4():
    return "这是Demo 4的内容"

@app.route('/robotcontrol')
def robotcontrol():
    return render_template('robot_control.html')

# === Demo1 API ===
@app.route('/api/demo1/start', methods=['POST'])
def demo1_start():
    return demo_controller.start_demo1()

@app.route('/api/demo1/stop', methods=['POST'])
def demo1_stop():
    return demo_controller.stop_demo1()

@app.route('/api/demo1/status', methods=['GET'])
def demo1_status():
    return demo_controller.get_demo1_status()

# === 机械臂控制API ===
@app.route('/api/robot/reset', methods=['POST'])
def robot_reset():
    return demo_controller.robot_reset()

@app.route('/api/gripper/reset', methods=['POST'])
def gripper_reset():
    return demo_controller.gripper_reset()

@app.route('/api/gripper/force_grasp', methods=['POST'])
def gripper_force_grasp():
    return demo_controller.gripper_force_grasp()

# === Camera API ===
@app.route('/api/camera/frame') #把下面的函数注册成接口路径 /api/camera/frame， 浏览器直接访问: http://localhost:5000/api/camera/frame
def camera_frame():
    """获取单帧图像"""
    try:
        frame = camera_reader.get_current_frame()
        if frame is None:
            return {'success': False, 'message': '无法获取相机画面'}
        
        # 调整图像大小以提高传输效率
        frame = cv2.resize(frame, (640, 480))
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'frame': frame_base64,
            'timestamp': time.time()
        }
    except Exception as e:
        return {'success': False, 'message': f'获取相机画面失败: {str(e)}'}

@app.route('/api/camera/stream')
def camera_stream():
    """视频流端点"""
    def generate_frames():
        while True:
            try:
                frame = camera_reader.get_current_frame()
                if frame is None:
                    continue
                
                # 调整图像大小
                frame = cv2.resize(frame, (640, 480))
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # 约30fps
            except Exception as e:
                print(f"视频流错误: {e}")
                break
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === ForceField API ===
@app.route('/api/forcefield/start', methods=['POST'])
def forcefield_start():
    """启动ForceField ROS桥接器"""
    try:
        success = forcefield_ros_bridge.start()
        if success:
            return {'success': True, 'message': 'ForceField ROS桥接器已启动'}
        else:
            return {'success': False, 'message': 'ForceField ROS桥接器启动失败'}
    except Exception as e:
        return {'success': False, 'message': f'ForceField启动异常: {str(e)}'}

@app.route('/api/forcefield/stop', methods=['POST'])
def forcefield_stop():
    """停止ForceField ROS桥接器"""
    try:
        forcefield_ros_bridge.stop()
        return {'success': True, 'message': 'ForceField ROS桥接器已停止'}
    except Exception as e:
        return {'success': False, 'message': f'ForceField停止异常: {str(e)}'}

@app.route('/api/forcefield/status', methods=['GET'])
def forcefield_status():
    """获取ForceField状态"""
    try:
        status = forcefield_ros_bridge.get_status()
        return {'success': True, 'status': status}
    except Exception as e:
        return {'success': False, 'message': f'获取ForceField状态失败: {str(e)}'}

# @app.route('/api/forcefield/frame')
# def forcefield_frame():
#     """获取ForceField单帧图像（拼接的三个图像）"""
#     try:
#         frame = forcefield_ros_bridge.get_current_frame()
#         if frame is None:
#             return {'success': False, 'message': '无法获取ForceField画面'}
        
#         # 调整图像大小以提高传输效率
#         frame = cv2.resize(frame, (960, 480))  # 保持宽高比，三个并排画面
        
#         # 编码为JPEG
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
#         frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
#         return {
#             'success': True,
#             'frame': frame_base64,
#             'timestamp': time.time()
#         }
#     except Exception as e:
#         return {'success': False, 'message': f'获取ForceField画面失败: {str(e)}'}

@app.route('/api/forcefield/three_frames')
def forcefield_three_frames():
    """获取ForceField三个独立图像"""
    try:
        frames = forcefield_ros_bridge.get_three_frames()
        if frames['tactile'] is None or frames['normal'] is None or frames['shear'] is None:
            return {'success': False, 'message': '无法获取完整的ForceField画面'}
        
        result = {'success': True, 'frames': {}, 'timestamp': time.time()}
        
        # 处理每个图像
        for frame_type, frame in frames.items():
            if frame is not None:
                # 调整图像大小
                frame = cv2.resize(frame, (320, 240))
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                result['frames'][frame_type] = frame_base64
        
        return result
    except Exception as e:
        return {'success': False, 'message': f'获取ForceField三个画面失败: {str(e)}'}

@app.route('/api/forcefield/stream')
def forcefield_stream():
    """ForceField视频流端点"""
    def generate_frames():
        while True:
            try:
                frame = forcefield_ros_bridge.get_current_frame()
                if frame is None:
                    continue
                
                # 调整图像大小
                frame = cv2.resize(frame, (960, 480))
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # 约30fps
            except Exception as e:
                print(f"ForceField视频流错误: {e}")
                break
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 禁用reloader避免双重初始化，但保持debug模式
    app.run(debug=True, use_reloader=False)
