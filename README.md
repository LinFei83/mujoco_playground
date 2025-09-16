# MuJoCo 游乐场

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

一个用于机器人学习研究和模拟到实物的GPU加速环境综合套件，基于[MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx)构建。

功能包括：

- 来自`dm_control`的经典控制环境。
- 四足和双足运动环境。
- 非预先操作和灵巧操作环境。
- 通过[Madrona-MJX](https://github.com/shacklettbp/madrona_mjx)提供基于视觉的支持。

有关更多详细信息，请查看项目[网站](https://playground.mujoco.org/)。

> [!NOTE]
> 我们现在支持使用MuJoCo MJX JAX实现和[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)在HEAD的实现进行训练。请参阅MuJoCo 3.3.5[发行说明](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-3-5-august-8-2025)中`MJX`部分的更多详细信息。

## 安装

你可以直接从PyPI安装MuJoCo Playground：

```sh
pip install playground
```

> [!WARNING]
> `playground`发行版可能依赖于`mujoco`和`warp-lang`的预发布版本，在这种情况下，你可以尝试`pip install playground
> --extra-index-url=https://py.mujoco.org
> --extra-index-url=https://pypi.nvidia.com/warp-lang/`。
> 如果仍有版本不匹配，请在GitHub上打开一个issue，并从源码安装。

### 从源码安装

> [!IMPORTANT]
> 需要Python 3.10或更高版本。

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. [安装uv](https://docs.astral.sh/uv/getting-started/installation/)，一个比`pip`更快的替代品
3. 创建虚拟环境：`uv venv --python 3.11`
4. 激活它：`source .venv/bin/activate`
5. 安装CUDA 12 jax：`uv pip install -U "jax[cuda12]"`
    * 验证GPU后端：`python -c "import jax; print(jax.default_backend())"` 应该打印gpu
6. 安装playground：`uv pip install -e ".[all]"`
7. 验证安装（并下载Menagerie）：`python -c "import mujoco_playground"`

#### Madrona-MJX（可选）

对于基于视觉的环境，请参考[Madrona-MJX](https://github.com/shacklettbp/madrona_mjx?tab=readme-ov-file#installation)仓库中的安装说明。

## 开始使用

### 基本教程

| Colab | 描述 |
|-------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb) | 使用DM Control Suite介绍Playground |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb) | 运动环境 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb) | 操作环境 |

### 基于视觉的教程（GPU Colab）

| Colab | 描述 |
|-------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1_t4.ipynb) | 从视觉训练CartPole（T4实例） |

### 本地运行时教程
*需要本地Madrona-MJX安装*

| Colab | 描述 |
|-------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb) | 从视觉训练CartPole |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb) | 机器人视觉操作 |

## 从CLI运行

> [!IMPORTANT]
> 假设从源码安装。

对于基本使用，导航到仓库目录并运行：
```bash
python learning/train_jax_ppo.py --env_name CartpoleBalance
```

### 训练可视化

要使用[rscope](https://github.com/Andrew-Luo1/rscope/tree/main)交互式查看训练过程中的轨迹，安装它（`pip install rscope`）并运行：

```
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
# 在另一个终端中
python -m rscope
```

## 常见问题

### 我如何贡献？

通过安装库并探索其功能开始！发现了一个bug？在issue tracker中报告。有兴趣贡献？如果你是一个有机器人经验的开发者，我们很乐意你的帮助——查看[贡献指南](CONTRIBUTING.md)了解更多详细信息。

### 可重现性 / GPU精度问题

使用NVIDIA Ampere架构GPU的用户（例如RTX 30和40系列）可能会由于JAX默认使用TF32进行矩阵乘法而在mujoco_playground中遇到可重现性[问题](https://github.com/google-deepmind/mujoco_playground/issues/86)。这种较低的精度可能会对RL训练稳定性产生不利影响。为了确保与使用完整float32精度的系统（例如Turing GPU）一致的行为，请在启动实验之前在终端中运行`export JAX_DEFAULT_MATMUL_PRECISION=highest`（或将其添加到`~/.bashrc`末尾）。

要使用论文中使用的相同学习脚本重现结果，请运行brax训练脚本，可从[这里](https://github.com/google/brax/blob/1ed3be220c9fdc9ef17c5cf80b1fa6ddc4fb34fa/brax/training/learner.py#L1)获取。使用`learning/train_jax_ppo.py`脚本时结果稍有不同，请参阅[这里](https://github.com/google-deepmind/mujoco_playground/issues/171)的issue了解更多上下文。

## 引用

如果您在科学作品中使用Playground，请按如下方式引用：

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## 许可证和免责声明

运动环境中粗糙地形使用的纹理来自[Polyhaven](https://polyhaven.com/a/rock_face)，并根据[CC0](https://creativecommons.org/public-domain/cc0/)许可。

本仓库中的所有其他内容根据Apache许可证2.0版许可。本仓库顶级[LICENSE](LICENSE)文件中提供了该许可证的副本。您也可以从https://www.apache.org/licenses/LICENSE-2.0获取。

这不是Google官方支持的产品。