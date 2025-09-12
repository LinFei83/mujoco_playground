# 训练强化学习智能体

在本目录中，我们演示如何使用[Brax](https://github.com/google/brax)和[RSL-RL](https://github.com/leggedrobotics/rsl_rl)从MuJoCo Playground环境中训练强化学习智能体。我们提供两个命令行入口点：`python train_jax_ppo.py`和`python train_rsl_rl.py`。

有关使用MuJoCo Playground进行强化学习的更详细教程，请参阅：

1. 使用DM Control Suite的Playground介绍 [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. 运动环境 [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. 操作环境 [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. 从视觉训练CartPole [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. 从视觉进行机器人操作 [![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## 使用brax PPO进行训练

要使用brax PPO进行训练，可以使用`train_jax_ppo.py`脚本。该脚本使用brax PPO算法在给定环境中训练智能体。

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

要使用像素观察训练基于视觉的策略：
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

使用`python train_jax_ppo.py --help`查看可能的选项和用法。日志和检查点保存在`logs`目录中。

## 使用RSL-RL进行训练

要使用RSL-RL进行训练，可以使用`train_rsl_rl.py`脚本。该脚本使用RSL-RL算法在给定环境中训练智能体。

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

要渲染结果策略的行为：
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

其中`run_name`是您要加载的运行名称（在训练运行开始时会在控制台中打印）。

日志和检查点保存在`logs`目录中。
