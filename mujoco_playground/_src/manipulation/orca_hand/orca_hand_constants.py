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
"""ORCA手的常量。"""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "orca_hand"
# RUBIK_SCENE_XML = ROOT_PATH / "xmls" / "scene" / "combined_rubik_scene.xml"
RUBIK_SCENE_XML = ROOT_PATH / "xmls" / "convex" / "combined_rubik_scene.xml"

# 基于combined_rubik_scene分析
NQ = 17  # 17个关节（1个手腕 + 16个手指关节）
NV = 17  # 17个自由度
NU = 17  # 17个执行器

# 基于场景分析的关节名称
JOINT_NAMES = [
    # 手腕
    "orcahand_right/right_wrist",
    # 拇指
    "orcahand_right/right_thumb_mcp",
    "orcahand_right/right_thumb_abd",
    "orcahand_right/right_thumb_pip",
    "orcahand_right/right_thumb_dip",
    # 食指
    "orcahand_right/right_index_abd",
    "orcahand_right/right_index_mcp",
    "orcahand_right/right_index_pip",
    # 中指
    "orcahand_right/right_middle_abd",
    "orcahand_right/right_middle_mcp",
    "orcahand_right/right_middle_pip",
    # 无名指
    "orcahand_right/right_ring_abd",
    "orcahand_right/right_ring_mcp",
    "orcahand_right/right_ring_pip",
    # 小指
    "orcahand_right/right_pinky_abd",
    "orcahand_right/right_pinky_mcp",
    "orcahand_right/right_pinky_pip",
]

# 基于场景分析的执行器名称
ACTUATOR_NAMES = [
    "orcahand_right/right_wrist_actuator",
    "orcahand_right/right_thumb_mcp_actuator",
    "orcahand_right/right_thumb_abd_actuator",
    "orcahand_right/right_thumb_pip_actuator",
    "orcahand_right/right_thumb_dip_actuator",
    "orcahand_right/right_index_abd_actuator",
    "orcahand_right/right_index_mcp_actuator",
    "orcahand_right/right_index_pip_actuator",
    "orcahand_right/right_middle_abd_actuator",
    "orcahand_right/right_middle_mcp_actuator",
    "orcahand_right/right_middle_pip_actuator",
    "orcahand_right/right_ring_abd_actuator",
    "orcahand_right/right_ring_mcp_actuator",
    "orcahand_right/right_ring_pip_actuator",
    "orcahand_right/right_pinky_abd_actuator",
    "orcahand_right/right_pinky_mcp_actuator",
    "orcahand_right/right_pinky_pip_actuator",
]

# 用于传感器读数的手指尖名称
FINGERTIP_NAMES = [
    "thumb_tip",
    "index_tip", 
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]

CUBE_JOINT_NAMES = [
    "rubik-v1.50/cube_tx",
    "rubik-v1.50/cube_ty",
    "rubik-v1.50/cube_tz",
    "rubik-v1.50/cube_rot",
]