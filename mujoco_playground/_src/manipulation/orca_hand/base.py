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
"""Base classes for ORCA hand."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand import orca_hand_constants as consts


def get_assets() -> Dict[str, bytes]:
  """Get all assets needed for ORCA hand environments."""
  assets = {}
  # Add ORCA hand specific assets
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "rubik")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "scene")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "Dexterous_Hand")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "stls")
  return assets


class OrcaHandEnv(mjx_env.MjxEnv):
  """Base class for ORCA hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  # Sensor readings for rubik's cube.

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    """Get palm position from the hand."""
    # Use the palm body position as reference
    palm_body_id = self._mj_model.body("orcahand_right/right_palm").id
    return data.xpos[palm_body_id]

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    """Get rubik's cube position."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xpos[cube_body_id]

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    """Get rubik's cube orientation as quaternion."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xquat[cube_body_id]

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    """Get rubik's cube linear velocity."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cvel[cube_body_id][:3]

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    """Get rubik's cube angular velocity."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cvel[cube_body_id][3:]

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    """Get rubik's cube angular acceleration."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.cacc[cube_body_id][3:]

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    """Get cube up vector (z-axis of cube frame)."""
    cube_body_id = self._mj_model.body("rubik-v1.50/middle").id
    return data.xmat[cube_body_id].reshape(3, 3)[:, 2]

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    """Get goal orientation for the cube."""
    # Use mocap body for goal orientation if available
    if self._mj_model.nmocap > 0:
      return data.mocap_quat[0]
    else:
      # Default to identity quaternion
      return jp.array([1.0, 0.0, 0.0, 0.0])

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    """Get goal up vector."""
    goal_quat = self.get_cube_goal_orientation(data)
    from mujoco.mjx._src import math
    goal_mat = math.quat_to_mat(goal_quat)
    return goal_mat[:, 2]  # z-axis

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the palm."""
    palm_pos = self.get_palm_position(data)
    fingertip_positions = []
    
    # Get positions of fingertip bodies
    fingertip_body_names = [
        "orcahand_right/right_thumb_dp",
        "orcahand_right/right_index_ip", 
        "orcahand_right/right_middle_ip",
        "orcahand_right/right_ring_ip",
        "orcahand_right/right_pinky_ip",
    ]
    
    for body_name in fingertip_body_names:
      try:
        body_id = self._mj_model.body(body_name).id
        tip_pos = data.xpos[body_id]
        relative_pos = tip_pos - palm_pos
        fingertip_positions.append(relative_pos)
      except KeyError:
        # If body doesn't exist, use zero vector
        fingertip_positions.append(jp.zeros(3))
    
    return jp.concatenate(fingertip_positions)

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
