from turtle import begin_poly
from camera_reader import CameraReader
import cv2
import rospy 
from env import create_env
import time


# 616.3925, -577.8794, 13.0501, 177.9379, -0.7046, -0.1232

if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    env = create_env("config.json", init_robot=True) 
    dobot = env.robot1["robot"]
    gripper = env.gripper
    current_pose = dobot.get_pose()
    print(current_pose)

    dobot.move_to_pose(680.0052, -261.4967, 10.7183, 179.1678, 1.6567, -2.9093, speed=7, acceleration=6) 
    gripper.control(680, 6, 20) 
    wait = rospy.Rate(1.0 / 3)
    wait.sleep()
    dobot.move_to_pose(680.0052, -261.4967, -37.7183, 179.1678, 1.6567, -2.9093, speed=7, acceleration=6)
    wait = rospy.Rate(1.0 / 2)
    wait.sleep()

    camera = CameraReader(camera_id=11)
    gripper.control(position=380, force=5, speed=12)
    # 闭合夹爪，直到camera检测到变化
    while True:
        frame = camera.get_current_frame()
        if camera.compare_with_init_image(frame, method='pixel_changes', threshold=40)['has_changes']:
            gripper.pause_movement()
            print("检测到变化")
            wait = rospy.Rate(1.0 / 1)
            wait.sleep()
            dobot.move_to_pose(680.0052, -261.4967, 10.7183, 179.1678, 1.6567, -2.9093, speed=7, acceleration=6) #将抓取物体抬升

            wait = rospy.Rate(1.0 / 3)
            wait.sleep()

            dobot.move_to_pose(616.3925, -577.8794, 20, 177.9379, -0.7046, -0.1232, speed=7, acceleration=6) #将抓取物体抬升


            wait = rospy.Rate(1.0 / 6)
            wait.sleep()

            dobot.move_to_pose(615, -577.8794, 7, 177.9379, -0.7046, -0.1232, speed=7, acceleration=6) #将抓取物体抬升

            break
    

    


