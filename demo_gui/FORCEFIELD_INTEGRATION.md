# ForceField 集成说明

## 概述

本集成将 `./wrc/sparsh/demo_forcefield.py` 中显示的三个并排画面（触觉图像、法向力、剪切力）成功集成到了 `./new_work/demo_gui/main.py` 的 `/demo1` 网页中。

## 新增文件

### 1. `forcefield_display.py`
- **功能**: ForceField显示模块，封装三个画面的生成和显示逻辑
- **主要类**: `ForceFieldDisplay`
- **核心方法**:
  - `initialize()`: 初始化ForceField系统
  - `start()`: 启动显示
  - `stop()`: 停止显示
  - `get_current_frame()`: 获取当前帧
  - `get_status()`: 获取状态信息

### 2. `test_forcefield_integration.py`
- **功能**: 测试ForceField集成是否正常工作
- **用法**: `python test_forcefield_integration.py`

## 修改文件

### 1. `main.py`
新增API端点：
- `POST /api/forcefield/start` - 启动ForceField显示
- `POST /api/forcefield/stop` - 停止ForceField显示
- `GET /api/forcefield/status` - 获取ForceField状态
- `GET /api/forcefield/frame` - 获取单帧图像
- `GET /api/forcefield/stream` - 视频流端点

### 2. `templates/demo1.html`
新增UI组件：
- ForceField三个并排画面显示区域
- 启动/停止ForceField按钮
- 状态信息显示
- 三个画面的说明标签

## 使用方法

### 1. 启动应用
```bash
cd /home/yimu/new_work/demo_gui
conda activate your_environment
python main.py
```

### 2. 访问网页
打开浏览器访问: `http://localhost:5000/demo1`

### 3. 使用ForceField功能
1. 点击"启动 Demo1"按钮
2. 系统会自动启动相机和ForceField显示
3. 在ForceField区域可以看到三个并排画面：
   - **左侧**: 触觉图像（实时触觉传感器数据）
   - **中间**: 法向力（垂直方向力场分析）
   - **右侧**: 剪切力（水平方向力场分析）

### 4. 独立控制
- 可以单独点击"启动ForceField"或"停止ForceField"按钮
- 可以单独控制相机和ForceField的启动/停止

## 技术实现

### 1. 画面生成逻辑
```python
# 获取当前触觉图像
current_tactile_image = self._get_current_tactile_image()

# 生成法向力和剪切力图像
im_normal, im_shear = self._generate_force_images(current_tactile_image)

# 水平拼接三个图像
im_h = cv2.hconcat([current_tactile_image, im_normal, im_shear])
```

### 2. 视频流传输
- 使用OpenCV的`cv2.imencode`将图像编码为JPEG
- 通过Flask的`Response`对象传输视频流
- 前端使用`<img>`标签的`src`属性接收流

### 3. 异步处理
- ForceField显示在独立线程中运行
- 使用线程锁保证数据安全
- 支持实时状态查询

## 配置要求

### 1. 环境依赖
- Python 3.x
- Flask
- OpenCV (cv2)
- NumPy
- PyTorch
- Sparsh相关模块

### 2. 文件路径
确保以下路径存在：
- `/home/yimu/wrc/sparsh/sparsh-gelsight-forcefield-decoder/gelsight_t1_forcefield_dino_vitbase_bg/config.yaml`
- `/home/yimu/wrc/sparsh/sparsh-gelsight-forcefield-decoder/gelsight_t1_forcefield_dino_vitbase_bg/checkpoints/`

### 3. 硬件要求
- GelSight设备ID: 10（可在`forcefield_display.py`中修改）

## 故障排除

### 1. 导入错误
如果遇到模块导入错误，检查：
- Sparsh项目路径是否正确
- 相关Python包是否已安装
- 环境变量是否正确设置

### 2. 画面不显示
检查：
- GelSight设备是否连接
- 设备ID是否正确
- 配置文件是否存在
- 模型检查点是否可用

### 3. 性能问题
- 调整图像分辨率（在`main.py`中修改`cv2.resize`参数）
- 调整JPEG质量（修改`cv2.IMWRITE_JPEG_QUALITY`参数）
- 调整帧率（修改`time.sleep`参数）

## 扩展功能

### 1. 添加更多画面
可以在`_generate_force_images`方法中添加更多力场分析画面

### 2. 实时参数调整
可以添加滑块控件来实时调整力场分析参数

### 3. 数据记录
可以添加数据记录功能，保存力场分析结果

## 注意事项

1. **模块化设计**: 新增的代码尽量不修改原有代码，保持向后兼容
2. **错误处理**: 所有关键操作都有异常处理
3. **资源管理**: 正确管理线程和硬件资源
4. **用户体验**: 提供清晰的状态反馈和错误信息

## 更新日志

- **v1.0**: 初始集成，支持三个并排画面显示
- 支持自动启动/停止
- 支持独立控制
- 支持实时状态监控
