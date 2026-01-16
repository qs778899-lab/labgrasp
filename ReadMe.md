# Manipulation framework based on Robotwin API

## 项目结构

config.json: cam, dobot等配置参数

GraspLibrary.json: target object的抓取配置参数

env.py: 软硬件对象初始化

level2_action.py: level2 actions 定义


calculate_grasp_pose_from_object_pose.py: 模块化封装子任务，类似level2 action (目前不再使用， 替换为level_action.py)


## 不同运行主文件的功能:

test_glassbar_grasp_main_tt_contact.py: 测试玻璃棒向下移动碰到漏斗壁是否及时停止

glassbar_grasp_main_tt_contact.py: 抓取玻璃棒调整姿态至倾斜向下，再向下移动触碰桌面

glassbar_grasp_main_v_contact.py: 抓取玻璃棒调整姿态至垂直向下，再向下移动触碰桌面







## 测试文件:

    test_changes_in_YIMU_monitor.py: 测试不同id对应的视触觉传感器设备是否连接以及能否感知形变

    compare_force_monitor.py: 对比力控和视触觉传感器对微小力的感知灵敏度

    T_base_cam1.py: 计算base坐标系下cam1(hand camera)的位姿 

    rotate_collection.py: 在固定高度上机械臂末端画一个圆收集不同视角的hand camera的照片








    


