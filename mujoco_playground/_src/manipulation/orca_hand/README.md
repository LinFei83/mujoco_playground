# 任务目标
参考mujoco_playground/_src/manipulation/leap_hand中的内容使用mujoco_playground框架实现灵巧手旋转locked魔方到指定姿态

## 1. 核心参考文件

**基础环境类**:

- `mujoco_playground/_src/manipulation/leap_hand/base.py` - LEAP Hand基础环境类
- `mujoco_playground/_src/manipulation/leap_hand/leap_hand_constants.py` - 常量定义

**现有任务实现**:

- `mujoco_playground/_src/manipulation/leap_hand/reorient.py` - 立方体重定向任务（最相关）
- `mujoco_playground/_src/manipulation/leap_hand/rotate_z.py` - Z轴旋转任务

**配置和注册**:

- `mujoco_playground/_src/manipulation/__init__.py` - 环境注册
- `mujoco_playground/config/manipulation_params.py` - 训练参数配置

**模型文件**:

- `mujoco_playground/_src/manipulation/leap_hand/xmls/` - LEAP Hand模型和场景文件

## 可用资源

### mujoco模型
- `mujoco_playground/_src/manipulation/orca_hand/xmls/rubik/rubik_locked.xml` - 这是一个内部没有自由度只有外形的locked魔方  

- `mujoco_playground/_src/manipulation/orca_hand/xmls/Dexterous_Hand/orcahand_right.mjcf` - 项目中需要使用的机械手

- `mujoco_playground/_src/manipulation/orca_hand/xmls/scene/combined_rubik_scene.xml` 将locked魔方与机械手,光照等场景结合后的场景, 将各个组件放置到对应的位置上

### 其他参考
`mujoco_playground/_src/manipulation/orca_hand/xmls/scene/combined_rubik_scene解析.txt` - 组合场景的关节与执行器分析


