# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""使用JAX在指定环境中训练PPO智能体。"""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX
import wandb


# 设置XLA标志以启用GPU上的Triton GEMM优化
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
# 禁用XLA内存预分配
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# 设置MuJoCo使用EGL渲染器
os.environ["MUJOCO_GL"] = "egl"

# 忽略来自brax的信息日志
logging.set_verbosity(logging.WARNING)

# 抑制警告

# 抑制来自JAX的运行时警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# 抑制来自JAX的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# 抑制来自absl的用户警告（被JAX和TensorFlow使用）
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"环境名称。可选值之一: {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX实现方式")
_VISION = flags.DEFINE_boolean("vision", False, "使用视觉输入")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "加载检查点的路径"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "实验名称的后缀")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "如果为真，仅使用模型进行游玩而不进行训练"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "使用Weights & Biases进行日志记录（在仅游玩模式下忽略）",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "使用TensorBoard进行日志记录（在仅游玩模式下忽略）"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "使用域随机化"
)
_SEED = flags.DEFINE_integer("seed", 1, "随机种子")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "时间步数量"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "训练后记录的视频数量"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "评估次数")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "奖励缩放因子")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "回合长度")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "归一化观察值"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "动作重复次数")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "展开长度")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "小批量数量"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "每批次更新次数"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "折扣因子")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "学习率")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "熵正则化系数")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "环境实例数量")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "评估环境实例数量"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "批量大小")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "最大梯度范数")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "PPO算法的剪裁参数ε"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "策略网络隐藏层大小",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "价值网络隐藏层大小",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "策略网络观察值键名"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "价值网络观察值键名")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "保存给rscope查看器的并行环境模拟数量",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "为rscope查看器运行确定性模拟",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "在策略更新之间运行评估模拟",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "是否记录训练指标并回调progress_fn函数。"
    "如果记录频率过高会显著降低训练速度",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "记录训练指标的间隔步数。如果训练速度变慢，请增加此值",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  """根据环境名称获取强化学习配置
  
  Args:
    env_name: 环境名称
    
  Returns:
    相应环境的PPO配置
    
  Raises:
    ValueError: 如果环境名称不在支持的列表中
  """
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  rscope查看器的回调函数，处理模拟数据并计算累积奖励
  
  所有数组形状为 (unroll_length, rscope_envs, ...)
  full_states: 包含'qpos', 'qvel', 'time', 'metrics'键的字典
  obs: 基于环境配置的nd.array或字典观察值
  rew: nd.array奖励值
  done: nd.array结束标志
  """
  # 计算每个回合的累积奖励，在遇到第一个结束标志时停止
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "收集到rscope模拟数据，平均奖励为"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """在指定环境中运行训练和评估。"""

  del argv

  # 加载环境配置
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value

  # 获取PPO算法参数
  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # 生成唯一的实验名称
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"实验名称: {exp_name}")

  # 设置日志目录
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"日志存储位置: {logdir}")

  # 如果需要，初始Weights & Biases日志工具
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # 如果需要，初始TensorBoard日志工具
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # 处理检查点加载
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # 转换为绝对路径
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"从以下路径恢复模型: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"从检查点恢复模型: {restore_checkpoint_path}")
  else:
    print("未提供检查点路径，不从检查点恢复")
    restore_checkpoint_path = None

  # 设置检查点目录
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"检查点路径: {ckpt_path}")

  # 保存环境配置
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # 准备训练参数
  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  # 选择网络工厂函数（基于是否使用视觉输入）
  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  # 如果启用域随机化，获取相应的随机化函数
  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  # 如果使用视觉输入，将环境包装为适合brax训练的形式
  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  # 确定评估环境数量
  num_eval_envs = (
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
  )

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  # 准备训练函数
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  # 记录训练时间
  times = [time.monotonic()]

  # 用于日志记录的进度函数
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # 记录到Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # 记录到TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  # 加载评估环境
  eval_env = None
  if not _VISION.value:
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  # 初始化策略参数函数
  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # 策略检查点的交互式可视化
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # 训练或加载模型
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("训练完成。")
  if len(times) > 1:
    print(f"JIT编译时间: {times[1] - times[0]}")
    print(f"训练时间: {times[-1] - times[1]}")

  print("开始推理...")

  # 创建推理函数
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # 运行评估模拟
  def do_rollout(rng, state):
    # 创建空轨迹数据结构
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    # 定义单步模拟函数
    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]  # 使用策略选择动作
      state = eval_env.step(state, act)  # 执行动作
      # 收集轨迹数据
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,  # 位置数据
          "data.qvel": state.data.qvel,  # 速度数据
          "data.time": state.data.time,  # 时间
          "data.ctrl": state.data.ctrl,  # 控制信号
          "data.mocap_pos": state.data.mocap_pos,  # 运动捕捉位置
          "data.mocap_quat": state.data.mocap_quat,  # 运动捕捉旋转
          "data.xfrc_applied": state.data.xfrc_applied,  # 外力
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    # 使用scan进行模拟
    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  # 生成多个随机种子并重置环境
  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  if _VISION.value:
    reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
  # 并行运行多个评估模拟
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  # 重新组织轨迹数据
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # 渲染并保存模拟视频
  render_every = 2  # 每隔几帧渲染一次
  fps = 1.0 / eval_env.dt / render_every  # 计算帧率
  print(f"渲染帧率: {fps}")
  # 设置场景选项
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False  # 关闭透明效果
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False  # 关闭扭力显示
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False  # 关闭接触力显示
  # 为每个模拟生成视频
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(f"rollout{i}.mp4", frames, fps=fps)
    print(f"模拟视频已保存为 'rollout{i}.mp4'.")


if __name__ == "__main__":
  app.run(main)
