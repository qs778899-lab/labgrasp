import cv2
import numpy as np
from Robotic_Arm.rm_robot_interface import *
from utils import GC, FloatSubscriber, CameraReader
from toolbox.arm_tb import ArmController
import time
from pynput.keyboard import Key, KeyCode, Listener
def on_key_press(key):
    global q_pressed
    if key == KeyCode.from_char('q'):
        q_pressed = True
        print("Q key pressed - will exit loop")


class ControDropper:
    def __init__(self, gripper):
        self.gripper = gripper
        
    def record_current_position(self):
        self.current_position = self.gripper.get_gripper_state()['position']

    def get_water(self):
        # 控制夹爪来通过滴管吸取水
        current_position = 130
        target_position = 0

        self.gripper.send_command(position=target_position, force=20, speed=30)

        input("按任意键继续")
        # insert the dropper into the water
        # change here to control water level
        self.gripper.send_command(position=max(current_position, 0), force=20, speed=7)

    def drop_water(self):
        # 控制夹爪来通过滴管吐水

        current_position = self.gripper.get_gripper_state()['position']
        target_position = max(current_position-20, 0)
        self.gripper.send_command(position=target_position, force=20, speed=7)







def catch_dropper():
    gripper.send_command(position=500.0, force=20, speed=30)

    time.sleep(1)
    camera_reader.reset_init()
    print("reset init image")
    gripper.send_command(position=100.0, force=20, speed=7)
    # 实时显示窗口
    print("按 'q' 键退出实时显示")
    while True:
        # 获取当前帧
        current_frame = camera_reader.get_current_frame()
        
        if current_frame is not None:
            # 检测像素变化
            change_result = camera_reader.compare_with_init_image(current_frame, method='pixel_changes', threshold=30)
            # print(f"变化像素: {change_result['changed_pixels']}")
            # print(f"变化百分比: {change_result['change_percentage']:.2f}%")
            if 'error' not in change_result:
                if change_result['has_changes'] and change_result["change_percentage"] > 0.001:
                    print("检测到变化，暂停运动")
                    gripper.pause_movement(force=20, speed=20)
                    dropper.record_current_position()
                    break
            
            # 调整图像大小
            height, width = current_frame.shape[:2]
            # 将图像缩小到原来的1/6大小
            resized_frame = cv2.resize(current_frame, (width//6, height//6))
            
            # 显示调整后的当前帧
            # cv2.imshow('Camera 10 - Live Feed', resized_frame)
        else:
            print("无法获取当前帧")
            break
        
        # 检查按键，按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    # camera_reader.release()

def control_volume(target):
    while True:
        print(fs.latest_value)
        if (target==fs.latest_value):
            break
        direction = np.sign(target-fs.latest_value)
        current_position = gripper.get_gripper_state()['position']
        target_position= current_position - direction * 4
        gripper.send_command(position=target_position, force=20, speed=7)
        time.sleep(2)








# 使用示例
if __name__ == "__main__":
    # 创建相机读取器实例
    fs = FloatSubscriber()
    camera_reader = CameraReader(camera_id=10)
    robo_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robo_arm.rm_create_robot_arm("192.168.1.18", 8080)
    arm = ArmController(robo_arm,handle)
    gripper = GC(arm=robo_arm, handle=handle, init=True)
    dropper = ControDropper(gripper)

    input("按任意键继续")
    
    while True:

        print("\nWhat would you like to do next?")
        print("1. catch ")
        # print("2. get water") 
        # print("3. drop water") 
        # print("4. drop water to target volume")
        # print("q. Quit")
        
        choice = input("Enter your choice (1/2/3/4/q): ").strip().lower()
        
        if choice == '1':
            listener = Listener(on_press=on_key_press)
            listener.start()
            while True:
                    global q_pressed
                    q_pressed = False  # Reset the flag
                    catch_dropper()
                    time.sleep(2)
                    gripper.send_command(position=1000)
                    time.sleep(3)
                    if q_pressed:
                        print("Exiting force monitoring loop due to 'q' key press")
                        gripper.send_command(position=800.0, force=20, speed=100)
                        break
        if choice == "0":
            gripper.send_command(position=1000)


    # 获取初始影像信息
    # init_info = camera_reader.get_init_image_info()
    # print("初始影像信息:", init_info)


