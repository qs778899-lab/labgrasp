from turtle import begin_poly
from camera_reader import CameraReader
import cv2
import rospy 
from env import create_env
import time




if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    env = create_env("config.json", init_robot=True) 
    dobot = env.robot1["robot"]
    gripper = env.gripper
    current_pose = dobot.get_pose()
    print(current_pose)

    # dobot.move_to_pose(584.5176, -253.6532, 10, -179.3843, -2.6657, -90.9127, speed=7, acceleration=6) 
    dobot.move_to_pose(677.4172, -437.3638, 10.4038, 179.8309, 0.0175, -3.1993, speed=7, acceleration=6)
    gripper.control(680, 6, 20) 
    wait = rospy.Rate(1.0 / 2)
    wait.sleep()
    # dobot.move_to_pose(584.5176, -253.6532, -39, -179.3843, -2.6657, -90.9127, speed=7, acceleration=6)
    dobot.move_to_pose(671.4172, -437.3638,  -37.7183, 179.8309, 0.0175, -3.1993, speed=7, acceleration=6)
    wait = rospy.Rate(1.0 / 2)
    wait.sleep()

    camera = CameraReader(camera_id=11)
    gripper.control(position=400, force=5, speed=8)
    # 闭合夹爪，直到camera检测到变化
    while True:
        frame = camera.get_current_frame()
        if camera.compare_with_init_image(frame, method='pixel_changes', threshold=50)['has_changes']:
            gripper.pause_movement()
            print("检测到变化")
            wait = rospy.Rate(1.0 / 1)
            wait.sleep()
            # dobot.move_to_pose(584.5176, -253.6532, 10, -179.3843, -2.6657, -90.9127, speed=7, acceleration=6) #将抓取物体抬升
            dobot.move_to_pose(677.4172, -437.3638, 10, 179.8309, 0.0175, -3.1993, speed=7, acceleration=6) #将抓取物体抬升

            wait = rospy.Rate(1.0 / 5)
            wait.sleep()
            gripper.control(650, 6, 20) 
            break
      
    


