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
"""Constants for ORCA hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "orca_hand"
RUBIK_SCENE_XML = ROOT_PATH / "xmls" / "scene" / "combined_rubik_scene.xml"

# Based on the combined_rubik_scene analysis
NQ = 17  # 17 joints (1 wrist + 16 finger joints)
NV = 17  # 17 DOF
NU = 17  # 17 actuators

# Joint names based on the scene analysis
JOINT_NAMES = [
    # wrist
    "orcahand_right/right_wrist",
    # thumb
    "orcahand_right/right_thumb_mcp",
    "orcahand_right/right_thumb_abd",
    "orcahand_right/right_thumb_pip",
    "orcahand_right/right_thumb_dip",
    # index finger
    "orcahand_right/right_index_abd",
    "orcahand_right/right_index_mcp",
    "orcahand_right/right_index_pip",
    # middle finger
    "orcahand_right/right_middle_abd",
    "orcahand_right/right_middle_mcp",
    "orcahand_right/right_middle_pip",
    # ring finger
    "orcahand_right/right_ring_abd",
    "orcahand_right/right_ring_mcp",
    "orcahand_right/right_ring_pip",
    # pinky finger
    "orcahand_right/right_pinky_abd",
    "orcahand_right/right_pinky_mcp",
    "orcahand_right/right_pinky_pip",
]

# Actuator names based on the scene analysis
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

# Fingertip names for sensor readings
FINGERTIP_NAMES = [
    "thumb_tip",
    "index_tip", 
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]
