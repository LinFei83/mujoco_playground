"""ORCA手的魔方重定向任务。"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.orca_hand import base as orca_hand_base
from mujoco_playground._src.manipulation.orca_hand import orca_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
  """ORCA手魔方重定向的默认配置。"""
  return config_dict.create(
      ctrl_dt=0.02, 
      sim_dt=0.01,  # 仿真步
      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.1,
      history_len=1,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
              position=0.5,
              termination=-100.0,
              hand_pose=-0.5,
              action_rate=-0.001,
              joint_vel=0.0,
              energy=-1e-3,
          ),
          success_reward=100.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
      impl='jax',
      nconmax=30 * 1024, # 接触约束
      njmax=128,
  )


class RubikReorient(orca_hand_base.OrcaHandEnv):
  """使用ORCA手将魔方重新定向以匹配目标方向。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.RUBIK_SCENE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    """初始化后设置。"""
    # 从关键帧获取初始配置（如果可用）
    try:
      home_key = self._mj_model.keyframe("home")
      self._init_q = jp.array(home_key.qpos, dtype=float)
      self._init_mpos = jp.array(home_key.mpos, dtype=float)
      self._init_mquat = jp.array(home_key.mquat, dtype=float)
    except:
      # 如果没有关键帧，使用默认值
      self._init_q = jp.zeros(self._mj_model.nq)
      self._init_mpos = jp.zeros(3 * self._mj_model.nmocap)
      self._init_mquat = jp.array([1.0, 0.0, 0.0, 0.0])
      if self._mj_model.nmocap > 0:
        self._init_mquat = jp.tile(self._init_mquat, (self._mj_model.nmocap, 1)).ravel()

    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    
    # 获取ORCA手的关节ID
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    
    # 获取魔方关节ID（用于自由关节）
    cube_joint_names = [
        "rubik-v1.50/cube_tx",
        "rubik-v1.50/cube_ty", 
        "rubik-v1.50/cube_tz",
        "rubik-v1.50/cube_rot",
    ]
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, cube_joint_names)
    
    # 获取物体和几何体ID
    self._cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    self._cube_geom_id = self._mj_model.geom("rubik-v1.50/middle").id
    self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
    
    # 默认手部姿态（中性位置）
    self._default_pose = jp.zeros(len(consts.JOINT_NAMES))

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """重置环境到初始状态。"""
    print("RubikReorient: 开始执行reset操作")
    # 使用固定的目标方向（单位四元数，无旋转）
    goal_quat = jp.array([1.0, 0.0, 0.0, 0.0])

    # 使用默认手部姿态
    q_hand = self._default_pose
    v_hand = jp.zeros(consts.NV)

    # 将魔方放置在手的前方的固定位置
    start_pos = jp.array([1.0, 0.87, 0.255])
    start_quat = jp.array([1.0, 0.0, 0.0, 0.0])  # 无旋转
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    # 组合所有关节位置和速度
    qpos = jp.zeros(self._mj_model.nq)
    qpos = qpos.at[self._hand_qids].set(q_hand)
    qpos = qpos.at[self._cube_qids].set(q_cube)
    
    qvel = jp.zeros(self._mj_model.nv)
    qvel = qvel.at[self._hand_dqids].set(v_hand)
    qvel = qvel.at[self._cube_qids[:6]].set(v_cube)

    # 创建初始数据
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        ctrl=q_hand,
        qvel=qvel,
        mocap_pos=self._init_mpos if self._mj_model.nmocap > 0 else None,
        mocap_quat=jp.array([goal_quat]) if self._mj_model.nmocap > 0 else None,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # 初始化信息字典（简化版）
    info = {
        "rng": rng,
        "step": 0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "qpos_error_history": jp.zeros(self._config.history_len * consts.NQ),
        "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
        "cube_ori_error_history": jp.zeros(self._config.history_len * 6),
        "goal_quat_dquat": jp.zeros(3),
        # 扰动相关（保持结构但不使用）
        "pert_wait_steps": jp.array([1000]),  # 设置很大的值避免扰动
        "pert_duration_steps": jp.array([1]),
        "pert_vel": jp.zeros(6),
        "pert_dir": jp.zeros(6, dtype=float),
        "last_pert_step": jp.array([-jp.inf], dtype=float),
    }

    # 初始化指标
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    print("RubikReorient: reset操作完成")
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """执行一步环境仿真。"""
    print("RubikReorient: 开始执行step操作")
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state, state.info["rng"])

    # 应用控制并执行物理仿真步骤
    delta = action * self._config.action_scale
    motor_targets = state.data.ctrl + delta
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    motor_targets = (
        self._config.ema_alpha * motor_targets
        + (1 - self._config.ema_alpha) * state.info["motor_targets"]
    )

    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # 检查是否成功
    ori_error = self._cube_orientation_error(data)
    success = ori_error < self._config.success_threshold
    state.info["steps_since_last_success"] = jp.where(
        success, 0, state.info["steps_since_last_success"] + 1
    )
    state.info["success_count"] = jp.where(
        success, state.info["success_count"] + 1, state.info["success_count"]
    )
    state.metrics["steps_since_last_success"] = state.info[
        "steps_since_last_success"
    ]
    state.metrics["success_count"] = state.info["success_count"]

    done = self._get_termination(data, state.info)
    obs = self._get_obs(data, state.info)

    # 计算奖励
    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    # 保持固定的目标方向（不进行随机化）
    # 目标始终是单位四元数（无旋转）
    fixed_goal_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    
    if self._mj_model.nmocap > 0:
      data = data.replace(mocap_quat=jp.array([fixed_goal_quat]))
    
    state.metrics["reward/success"] = success.astype(float)
    reward += success * self._config.reward_config.success_reward

    # 更新信息和指标
    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    print("RubikReorient: step操作完成")
    return state

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """检查终止条件。"""
    print("RubikReorient: 开始检查终止条件。")
    del info  # 未使用
    # 如果魔方下落到某个高度以下则终止
    fall_termination = self.get_cube_position(data)[2] < 0.1
    nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    print("RubikReorient: 检查终止条件完成。")
    return fall_termination | nans

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    """获取观测值。"""
    print("RubikReorient: 开始获取观测值。")
    # 手部关节角度
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    # 关节位置误差历史
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], consts.NQ)
        .at[:consts.NQ]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    def _get_cube_pose(data: mjx.Data) -> jax.Array:
      """返回（可能包含噪声的）魔方姿态(xyz,wxyz)。"""
      cube_pos = self.get_cube_position(data)
      cube_quat = self.get_cube_orientation(data)
      info["rng"], pos_rng, ori_rng = jax.random.split(info["rng"], 3)
      noisy_cube_quat = mjx._src.math.normalize(
          cube_quat
          + jax.random.normal(ori_rng, shape=(4,))
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_ori
      )
      noisy_cube_pos = (
          cube_pos
          + (2 * jax.random.uniform(pos_rng, shape=cube_pos.shape) - 1)
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_pos
      )
      return jp.concatenate([noisy_cube_pos, noisy_cube_quat])

    # 包含噪声的魔方姿态
    noisy_pose = _get_cube_pose(data)
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    rand_quat = orca_hand_base.uniform_quat(key1)
    rand_pos = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
    rand_pose = jp.concatenate([rand_pos, rand_quat])
    m = self._config.obs_noise.level * jax.random.bernoulli(
        key3, self._config.obs_noise.random_ori_injection_prob
    )
    noisy_pose = noisy_pose * (1 - m) + rand_pose * m

    # 魔方位置误差历史
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - noisy_pose[:3]
    cube_pos_error_history = (
        jp.roll(info["cube_pos_error_history"], 3).at[:3].set(cube_pos_error)
    )
    info["cube_pos_error_history"] = cube_pos_error_history

    # 魔方方向误差历史
    goal_quat = self.get_cube_goal_orientation(data)
    quat_diff = mjx._src.math.quat_mul(
        noisy_pose[3:], mjx._src.math.quat_inv(goal_quat)
    )
    xmat_diff = mjx._src.math.quat_to_mat(quat_diff).ravel()[3:]
    cube_ori_error_history = (
        jp.roll(info["cube_ori_error_history"], 6).at[:6].set(xmat_diff)
    )
    info["cube_ori_error_history"] = cube_ori_error_history

    # 用于评价器的无损魔方姿态
    cube_pos_error_uncorrupted = palm_pos - self.get_cube_position(data)
    cube_quat_uncorrupted = self.get_cube_orientation(data)
    quat_diff_uncorrupted = math.quat_mul(
        cube_quat_uncorrupted, math.quat_inv(goal_quat)
    )
    xmat_diff_uncorrupted = math.quat_to_mat(quat_diff_uncorrupted).ravel()[3:]

    state = jp.concatenate([
        noisy_joint_angles,  # NQ
        qpos_error_history,  # NQ * history_len
        cube_pos_error_history,  # 3 * history_len
        cube_ori_error_history,  # 6 * history_len
        info["last_act"],  # NU
    ])

    privileged_state = jp.concatenate([
        state,
        data.qpos[self._hand_qids],
        data.qvel[self._hand_dqids],
        self.get_fingertip_positions(data),
        cube_pos_error_uncorrupted,
        xmat_diff_uncorrupted,
        self.get_cube_linvel(data),
        self.get_cube_angvel(data),
        info["pert_dir"],
        data.xfrc_applied[self._cube_body_id],
    ])
    print("RubikReorient: 获取观测值完成。")
    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  # 奖励项

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    """计算奖励组件。"""
    print("RubikReorient: 开始计算奖励组件。")
    del done, metrics  # 未使用

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pose_mse = jp.linalg.norm(palm_pos - cube_pos)
    cube_pos_reward = reward.tolerance(
        cube_pose_mse, (0, 0.02), margin=0.05, sigmoid="linear"
    )

    terminated = self._get_termination(data, info)

    hand_pose_reward = jp.sum(
        jp.square(data.qpos[self._hand_qids] - self._default_pose)
    )
    print("RubikReorient: 计算奖励组件完成。")
    return {
        "orientation": self._reward_cube_orientation(data),
        "position": cube_pos_reward,
        "termination": terminated,
        "hand_pose": hand_pose_reward,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "joint_vel": self._cost_joint_vel(data),
        "energy": self._cost_energy(
            data.qvel[self._hand_dqids], data.actuator_force
        ),
    }

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    """基于关节速度和执行器力的能量成本。"""
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cube_orientation_error(self, data: mjx.Data) -> jax.Array:
    """计算魔方方向误差。"""
    cube_ori = self.get_cube_orientation(data)
    cube_goal_ori = self.get_cube_goal_orientation(data)
    quat_diff = math.quat_mul(cube_ori, math.quat_inv(cube_goal_ori))
    quat_diff = math.normalize(quat_diff)
    return 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), a_max=1.0))

  def _reward_cube_orientation(self, data: mjx.Data) -> jax.Array:
    """魔方方向奖励。"""
    ori_error = self._cube_orientation_error(data)
    return reward.tolerance(ori_error, (0, 0.2), margin=jp.pi, sigmoid="linear")

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """动作变化率成本，鼓励平滑的动作。"""
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2

  def _cost_joint_vel(self, data: mjx.Data) -> jax.Array:
    """关节速度成本。"""
    max_velocity = 5.0
    vel_tolerance = 1.0
    hand_qvel = data.qvel[self._hand_dqids]
    return jp.sum((hand_qvel / (max_velocity - vel_tolerance)) ** 2)

  # 扰动
  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    """如果启用，对魔方应用扰动。"""
    def gen_dir(rng: jax.Array) -> jax.Array:
      directory = jax.random.normal(rng, (6,))
      return directory / jp.linalg.norm(directory)

    def get_xfrc(
        state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
      force = (
          u_t
          * self._cube_mass
          * state.info["pert_vel"]
          / (state.info["pert_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
      return xfrc_applied

    step, last_pert_step = state.info["step"], state.info["last_pert_step"]
    start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
    start_pert &= step != 0  # 在回合开始时不进行扰动
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert_interval = duration < state.info["pert_duration_steps"]

    pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
    xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

    state.info["pert_dir"] = pert_dir
    state.info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc)
    return state.replace(data=data)


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """ORCA手魔方任务的领域随机化。"""
pass