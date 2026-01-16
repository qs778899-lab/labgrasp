from turtle import begin_poly
from camera_reader import CameraReader
import cv2
import rospy 
from env import create_env
import time
from pynput import keyboard








# camera = CameraReader(camera_id=10)
# frame = camera.get_current_frame()
# cv2.imshow("Camera Preview", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def keyboard_control(dobot, camera):
    """使用 pynput 监听键盘控制机械臂"""
    running = True
    current_key = None

    def on_press(key):
        nonlocal current_key
        try:
            current_key = key.char
        except AttributeError:
            current_key = None

    def on_release(key):
        nonlocal current_key, running
        current_key = None
        if key == keyboard.Key.esc:
            running = False
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("键盘控制已启动: w/s=Y轴, a/d=X轴, r/f=Z轴, q=退出")

    while running:
        if current_key is None:
            time.sleep(0.05)
            continue

        compare_frame = camera.get_current_frame().copy()
        move = False
        camera.reset_init()
        if current_key == 'w':
            current_pose = dobot.get_pose()
            dobot.move_to_pose(current_pose[0], current_pose[1]+5, current_pose[2], current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 's':
            current_pose = dobot.get_pose()
            dobot.move_to_pose(current_pose[0], current_pose[1]-5, current_pose[2], current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 'a':
            current_pose = dobot.get_pose()
            dobot.move_to_pose(current_pose[0]-5, current_pose[1], current_pose[2], current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 'd':
            current_pose = dobot.get_pose()

            dobot.move_to_pose(current_pose[0]+5, current_pose[1], current_pose[2], current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 'r':
            current_pose = dobot.get_pose()
            dobot.move_to_pose(current_pose[0], current_pose[1], current_pose[2]+5, current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 'f':
            current_pose = dobot.get_pose()
            dobot.move_to_pose(current_pose[0], current_pose[1], current_pose[2]-5, current_pose[3], current_pose[4], current_pose[5],speed=10)
            move = True
        elif current_key == 'q':
            break

        current_key = None  # 清除按键，避免重复触发

        begin_time = time.time()
        while move:
            if time.time() - begin_time > 1:
                break
            current_frame = camera.get_current_frame()
            if camera.compare_with_init_image(current_frame, method='pixel_changes', threshold=5)['has_changes']:
                print("检测到变化")
                dobot.stop()
                print("============================")
                break
            else:
                print("未检测到变化")
    listener.stop()
    print("键盘控制已退出")


if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    env = create_env("config.json", init_robot=True) 
    dobot = env.robot1["robot"]
    gripper = env.gripper
    input("break")
    current_pose = dobot.get_pose()
    print(current_pose)

    camera = CameraReader(camera_id=10)
    gripper.control(position=300, force=100, speed=10)
    # 闭合夹爪，直到camera检测到变化
    while True:
        frame = camera.get_current_frame()
        if camera.compare_with_init_image(frame, method='pixel_changes', threshold=50)['has_changes']:
            gripper.pause_movement()
            print("检测到变化")
            break
        else:
            gripper.read_current_position()==300
            break




    input("break")

    keyboard_control(dobot,camera)

    input("break")
