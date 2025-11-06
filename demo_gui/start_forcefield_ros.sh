#!/bin/bash

# ForceField ROS启动脚本
# 用于启动ForceField ROS发布器和Web应用

echo "🚀 启动ForceField ROS系统..."

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

# 启动ForceField ROS发布器
echo "📡 启动ForceField ROS发布器..."
cd /home/yimu/wrc/sparsh
conda activate tactile
python forcefield_ros_publisher.py &
FORCEFIELD_PID=$!

# 等待发布器启动
sleep 5

# 启动Web应用
echo "🌐 启动Web应用..."
cd /home/yimu/new_work/demo_gui
conda activate py311
python main.py &
WEB_PID=$!

echo "✅ 系统启动完成！"
echo "📋 进程信息:"
echo "   - ForceField发布器 PID: $FORCEFIELD_PID"
echo "   - Web应用 PID: $WEB_PID"
echo ""
echo "🌐 访问地址: http://localhost:5000/demo1"
echo "📡 ROS话题: /forcefield/combined_image"
echo ""
echo "💡 按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo "🛑 停止服务..."; kill $FORCEFIELD_PID $WEB_PID; exit 0' INT
wait
