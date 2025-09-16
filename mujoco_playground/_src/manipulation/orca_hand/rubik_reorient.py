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
        ctrl_dt=0.05,  # 控制时间步长（秒）
        sim_dt=0.01,  # 仿真时间步长（秒）
        action_scale=0.5,  # 动作缩放因子
        action_repeat=1,  # 动作重复次数
        ema_alpha=1.0,  # EMA（指数移动平均）平滑因子
        episode_length=1000,  # 每个episode的最大步骤数
        success_threshold=0.1,  # 成功判定阈值（方向误差）
        history_len=1,  # 观测历史长度
        obs_noise=config_dict.create(  # 观测噪声配置
            level=1.0,  # 噪声水平
            scales=config_dict.create(
                joint_pos=0.05,  # 关节位置噪声标准差
                cube_pos=0.02,  # 魔方位置噪声标准差
                cube_ori=0.1,  # 魔方方向噪声标准差
            ),
            random_ori_injection_prob=0.0,  # 随机方向注入概率
        ),
        reward_config=config_dict.create(  # 奖励配置
            scales=config_dict.create(
                orientation=5.0,  # 方向奖励权重
                position=0.5,  # 位置奖励权重
                termination=-100.0,  # 终止惩罚权重
                hand_pose=-0.5,  # 手部姿态惩罚权重
                action_rate=-0.001,  # 动作变化率惩罚权重
                joint_vel=0.0,  # 关节速度惩罚权重（未使用）
                energy=-1e-3,  # 能量消耗惩罚权重
            ),
            success_reward=100.0,  # 成功奖励系数
        ),
        pert_config=config_dict.create(  # 扰动配置
            enable=False,  # 是否启用扰动
            linear_velocity_pert=[0.0, 3.0],  # 线性速度扰动范围[m/s]
            angular_velocity_pert=[0.0, 0.5],  # 角速度扰动范围[rad/s]
            pert_duration_steps=[1, 100],  # 扰动持续步骤数范围
            pert_wait_steps=[60, 150],  # 扰动等待步骤数范围
        ),
        impl="jax",  # 实现方式（jax）
        nconmax=30 * 1024,  # 最大接触约束数
        njmax=128,  # 最大关节数
    )


class RubikReorient(orca_hand_base.OrcaHandEnv):
    """使用ORCA手将魔方重新定向以匹配目标方向。"""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        # 可选的配置覆盖参数
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
            # 从xml文件中提取场景中关节的初始位置等
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
                self._init_mquat = jp.tile(
                    self._init_mquat, (self._mj_model.nmocap, 1)
                ).ravel()

        # 提取执行器（actuators）的控制范围（control range）
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

        # 获取ORCA手的关节ID
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)

        # 获取魔方关节ID（用于自由关节）
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, consts. CUBE_JOINT_NAMES)

        # 获取物体和几何体ID 
        # 魔方中boby与geom中的名字都是"rubik-v1.50/middle"
        self._cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
        self._cube_geom_id = self._mj_model.geom("rubik-v1.50/middle").id
        self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]

        # 默认手部姿态（中性位置）
        self._default_pose = jp.zeros(len(consts.JOINT_NAMES))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """重置环境到初始状态。"""
        print("RubikReorient: 执行reset操作")
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
            qpos=qpos, # 关节位置向量
            ctrl=q_hand, # 控制信号
            qvel=qvel, # 关节速度向量
            # mocap 物体的位置和姿态（如果有）
            mocap_pos= None,
            mocap_quat= None,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # 初始化信息字典
        info = {
            "rng": rng,  # 随机数生成器，用于复现训练过程
            "step": 0,  # 当前回合已进行的步数
            "steps_since_last_success": 0,  # 距离上一次成功过去了多少步
            "success_count": 0,  # 当前回合成功任务的总次数
            "last_act": jp.zeros(self.mjx_model.nu),  # 上一步的动作，用于动作平滑度计算
            "last_last_act": jp.zeros(self.mjx_model.nu),  # 上上一步的动作，用于动作平滑度计算
            "motor_targets": data.ctrl,  # 当前发送给执行器的控制目标值
            "qpos_error_history": jp.zeros(self._config.history_len * consts.NQ),  # 关节位置误差历史，用于观测
            "cube_pos_error_history": jp.zeros(self._config.history_len * 3),  # 魔方位置误差历史
            "cube_ori_error_history": jp.zeros(self._config.history_len * 6),  # 魔方姿态误差历史
            "goal_quat_dquat": jp.zeros(3),  # 目标姿态四元数与当前姿态的差值变化率
            "pert_wait_steps": jp.array([1000]),  # 设置很大的值避免扰动1000回合一次
            "pert_duration_steps": jp.array([1]),  # 扰动持续的步数
            "pert_vel": jp.zeros(6),  # 扰动速度向量（线性+角速度）
            "pert_dir": jp.zeros(6, dtype=float),  # 扰动方向
            "last_pert_step": jp.array([-jp.inf], dtype=float),  # 上一次扰动发生的步数
        }

        # 为每个奖励指标进行初始化
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["reward/success"] = jp.zeros((), dtype=float)
        metrics["steps_since_last_success"] = 0
        metrics["success_count"] = 0

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """执行一步环境仿真。"""
        # 是否启用扰动
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state, state.info["rng"])

        # 应用控制并执行物理仿真步骤
        # 对智能体输出的原始动作进行缩放。
        delta = action * self._config.action_scale
        # 增量式控制计算新的目标电机控制值
        motor_targets = state.data.ctrl + delta
        # 控制值限幅
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        # 根据ema_alpha进行指数移动平均(EMA)平滑处理。
        motor_targets = (
            self._config.ema_alpha * motor_targets
            + (1 - self._config.ema_alpha) * state.info["motor_targets"]
        )
        
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self. n_substeps)
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
        return state

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """检查终止条件。"""
        del info  # 未使用
        # 如果魔方下落到某个高度以下则终止
        fall_termination = self.get_cube_position(data)[2] < 0.1
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        return fall_termination | nans

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        """获取观测值。"""
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
            .at[: consts.NQ]
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

        state = jp.concatenate(
            [
                noisy_joint_angles,  # NQ
                qpos_error_history,  # NQ * history_len
                cube_pos_error_history,  # 3 * history_len
                cube_ori_error_history,  # 6 * history_len
                info["last_act"],  # NU
            ]
        )

        privileged_state = jp.concatenate(
            [
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
            ]
        )

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

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
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
    mj_model = RubikReorient().mj_model
    cube_geom_id = mj_model.geom("rubik-v1.50/middle").id
    cube_body_id = mj_model.body("rubik-v1.50/middle").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)

    # 获取手部物体名称（针对ORCA手简化）
    hand_body_names = [
        "orcahand_right/right_palm",
        "orcahand_right/right_thumb_mp",
        "orcahand_right/right_thumb_pp",
        "orcahand_right/right_thumb_ip",
        "orcahand_right/right_thumb_dp",
        "orcahand_right/right_index_mp",
        "orcahand_right/right_index_pp",
        "orcahand_right/right_index_ip",
        "orcahand_right/right_middle_mp",
        "orcahand_right/right_middle_pp",
        "orcahand_right/right_middle_ip",
        "orcahand_right/right_ring_mp",
        "orcahand_right/right_ring_pp",
        "orcahand_right/right_ring_ip",
        "orcahand_right/right_pinky_mp",
        "orcahand_right/right_pinky_pp",
        "orcahand_right/right_pinky_ip",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)

        # 缩放魔方质量：*U(0.8, 1.2)
        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        # 抖动qpos0：+U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[hand_qids].set(
            qpos0[hand_qids]
            + jax.random.uniform(key, shape=(consts.NQ,), minval=-0.05, maxval=0.05)
        )

        # 缩放静摩擦：*U(0.9, 1.1)
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        # 缩放电枢：*U(1.0, 1.05)
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(consts.NQ,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        # 缩放所有连杆质量：*U(0.9, 1.1)
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        # 关节刚度：*U(0.8, 1.2)
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # 关节阻尼：*U(0.8, 1.2)
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (consts.NQ,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "dof_damping": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    model = model.tree_replace(
        {
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes
