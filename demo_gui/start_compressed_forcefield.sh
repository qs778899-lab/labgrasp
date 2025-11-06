#!/bin/bash

# ForceField压缩图像ROS启动脚本
# 用于启动ForceField压缩图像发布器和接收器

echo "🚀 启动ForceField压缩图像ROS系统..."

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS环境未设置，请先source ROS环境"
    echo "💡 请运行: source /opt/ros/noetic/setup.bash"
    exit 1
fi

echo "✅ ROS环境: $ROS_DISTRO"

# 启动roscore（如果未运行）
if ! pgrep -x "roscore" > /dev/null; then
    echo "🔄 启动roscore..."
    roscore &
    sleep 3
fi

# 启动ForceField压缩图像发布器
echo "📡 启动ForceField压缩图像发布器..."
cd /home/yimu/wrc/sparsh
conda activate tactile
python forcefield_ros_publisher.py &
FORCEFIELD_PID=$!

# 等待发布器启动
sleep 5

# 启动压缩图像接收器
echo "📥 启动压缩图像接收器..."
cd /home/yimu/new_work/demo_gui
conda activate py311
python forcefield_compressed_receiver_test.py &
RECEIVER_PID=$!

echo "✅ 压缩图像系统启动完成！"
echo "📋 进程信息:"
echo "   - ForceField压缩发布器 PID: $FORCEFIELD_PID"
echo "   - 压缩图像接收器 PID: $RECEIVER_PID"
echo ""
echo "📡 压缩图像话题:"
echo "   - /forcefield/tactile_image/compressed"
echo "   - /forcefield/normal_force/compressed"
echo "   - /forcefield/shear_force/compressed"
echo ""
echo "💡 按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo "🛑 停止服务..."; kill $FORCEFIELD_PID $RECEIVER_PID; exit 0' INT
wait
