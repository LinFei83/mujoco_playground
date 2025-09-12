"""ORCA手部的基础类。

本模块包含ORCA手部环境的基础实现，提供了手部操作任务的通用功能，
包括传感器读取、状态获取和环境配置等。
"""

from typing import Any, Dict, Optional, Union

# 导入路径和文件操作工具
from etils import epath
# JAX数值计算库
import jax
import jax.numpy as jp
# 配置管理
from ml_collections import config_dict
# MuJoCo物理仿真引擎
import mujoco
from mujoco import mjx

# 项目内部模块
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand import orca_hand_constants as consts


def get_assets() -> Dict[str, bytes]:
  """获取ORCA手部环境所需的所有资源文件。
  
  Returns:
    包含所有必需资源文件的字典，键为文件路径，值为文件内容的字节数据。
  """
  assets = {}
  # 添加ORCA手部特定的资源文件
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")  # XML模型文件
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "rubik")  # 魔方模型
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "scene")  # 场景文件
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "Dexterous_Hand")  # 灵巧手模型
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "stls")  # STL几何文件
  return assets


class OrcaHandEnv(mjx_env.MjxEnv):
  """ORCA手部环境的基础类。
  
  该类继承自MjxEnv，为ORCA手部操作任务提供通用的环境接口和功能。
  包括手部传感器读取、魔方状态获取等功能。
  """

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    """初始化ORCA手部环境。
    
    Args:
      xml_path: MuJoCo XML模型文件的路径
      config: 环境配置参数
      config_overrides: 可选的配置覆盖参数
    """
    super().__init__(config, config_overrides)
    
    # 加载模型资源文件
    self._model_assets = get_assets()
    
    # 从 XML字符串创建 MuJoCo 模型
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    
    # 设置仿真时间步长
    self._mj_model.opt.timestep = self._config.sim_dt

    # 设置离屏渲染分辨率（4K分辨率）
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # 将 MuJoCo 模型转换为 MJX 模型（GPU加速）
    # 转换 MuJoCo 模型为 MJX 模型
    print(f"开始将 MuJoCo 模型转换为 MJX 模型: {xml_path}")
    print(f"模型统计信息 - Bodies: {self._mj_model.nbody}, "
          f"Joints: {self._mj_model.njnt}, "
          f"Actuators: {self._mj_model.nu}, "
          f"Implementation: {self._config.impl}")
    
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    
    print("✅ MJX 模型转换成功完成!")
    print(f"MJX 模型信息 - DOF: {self._mjx_model.nq}, "
          f"Velocities: {self._mjx_model.nv}, "
          f"Controls: {self._mjx_model.nu}")
    print(f"JAX 设备: {self._mjx_model.impl}")
    self._xml_path = xml_path

  # 魔方的传感器读取方法

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    """获取手掌的位置。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      手掌在世界坐标系中的位置 (3D向量)
    """
    # 使用手掌物体位置作为参考
    palm_body_id = self._mj_model.body("orcahand_right/right_palm").id
    return data.xpos[palm_body_id]

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    """获取魔方的位置。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方在世界坐标系中的位置 (3D向量)
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xpos[cube_body_id]

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    """获取魔方的朝向（四元数形式）。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方的朝向四元数 [w, x, y, z]
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xquat[cube_body_id]

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    """获取魔方的线速度。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方在世界坐标系中的线速度 (3D向量)
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cvel[cube_body_id][:3]

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    """获取魔方的角速度。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方在世界坐标系中的角速度 (3D向量)
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cvel[cube_body_id][3:]

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    """获取魔方的角加速度。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方在世界坐标系中的角加速度 (3D向量)
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cacc[cube_body_id][3:]

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    """获取魔方的向上向量（魔方坐标系的z轴）。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方坐标系的z轴单位向量 (3D向量)
    """
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xmat[cube_body_id].reshape(3, 3)[:, 2]

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    """获取魔方的目标朝向。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      魔方目标朝向的四元数 [w, x, y, z]
    """
    # 如果可用，使用mocap物体作为目标朝向
    if self._mj_model.nmocap > 0:
      return data.mocap_quat[0]
    else:
      # 默认使用单位四元数
      return jp.array([1.0, 0.0, 0.0, 0.0])

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    """获取目标向上向量。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      目标朝向的z轴单位向量 (3D向量)
    """
    goal_quat = self.get_cube_goal_orientation(data)
    from mujoco.mjx._src import math
    goal_mat = math.quat_to_mat(goal_quat)
    return goal_mat[:, 2]  # z轴

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """获取指尖相对于手掌的位置。
    
    Args:
      data: MJX仿真数据
      
    Returns:
      所有指尖相对于手掌的位置，拼接成一个向量 (15D: 5个手指 x 3D位置)
    """
    palm_pos = self.get_palm_position(data)
    fingertip_positions = []
    
    # 获取指尖物体的位置
    fingertip_body_names = [
        "orcahand_right/right_thumb_dp",    # 拇指指尖
        "orcahand_right/right_index_ip",    # 食指指尖
        "orcahand_right/right_middle_ip",   # 中指指尖
        "orcahand_right/right_ring_ip",     # 无名指指尖
        "orcahand_right/right_pinky_ip",    # 小指指尖
    ]
    
    for body_name in fingertip_body_names:
      try:
        body_id = self._mj_model.body(body_name).id
        tip_pos = data.xpos[body_id]
        relative_pos = tip_pos - palm_pos  # 计算相对位置
        fingertip_positions.append(relative_pos)
      except KeyError:
        # 如果物体不存在，使用零向量
        fingertip_positions.append(jp.zeros(3))
    
    return jp.concatenate(fingertip_positions)

  # 属性访问器

  @property
  def xml_path(self) -> str:
    """获取XML模型文件路径。
    
    Returns:
      XML模型文件的路径字符串
    """
    return self._xml_path

  @property
  def action_size(self) -> int:
    """获取动作空间的维度。
    
    Returns:
      动作空间的维度数（机器人关节数量）
    """
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    """获取MuJoCo模型对象。
    
    Returns:
      MuJoCo物理模型对象，用于CPU仿真
    """
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    """获取MJX模型对象。
    
    Returns:
      MJX模型对象，用于GPU加速仿真
    """
    return self._mjx_model


def uniform_quat(rng: jax.Array) -> jax.Array:
  """从均匀分布中生成随机四元数。
  
  使用Marsaglia算法生成均匀分布的单位四元数，用于随机旋转。
  
  Args:
    rng: JAX随机数生成器状态
    
  Returns:
    均匀分布的单位四元数 [x, y, z, w]
  """
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),  # x分量
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),  # y分量
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),      # z分量
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),      # w分量
  ])
