# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from unitree_rl_lab (unitree_rl_lab/assets/robots/unitree.py, which
# carried the BSD-3-Clause header above; the upstream repository as a whole is
# licensed under Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: The three base config classes are verbatim ports of
# UnitreeArticulationCfg / UnitreeUsdFileCfg / UnitreeUrdfFileCfg with
# vendor-neutral names and a project-local /tmp staging directory.
# ``build_articulation_cfg`` is original yanshi-rl-lab work
# (Copyright (c) 2026 Jeff Liu, MIT).

"""Isaac Lab articulation configs built from a :class:`RobotProfile`.

This module may import Isaac Lab (unlike ``profile.py``, which must stay pure
Python).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from yanshi_rl_lab.robots.profile import RobotProfile

# Staging area used by ``replace_asset`` (upstream used /tmp/IsaacLab/unitree_rl_lab).
_TMP_ASSET_DIR = "/tmp/IsaacLab/yanshi_rl_lab"


@configclass
class RobotArticulationCfg(ArticulationCfg):
    """Articulation config carrying the real-robot SDK joint ordering."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class RobotUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


@configclass
class RobotUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When the collisions need changing, place the modified URDF file separately in this
        repository and let ``meshes_dir`` be provided by the vendor asset tree. This function
        auto-constructs a complete ``robot_description`` file structure under ``/tmp``.
        Note: mesh references inside the URDF must be at the same directory level as the URDF.
        """
        tmp_meshes_dir = f"{_TMP_ASSET_DIR}/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs(_TMP_ASSET_DIR, exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = f"{_TMP_ASSET_DIR}/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


def build_articulation_cfg(profile: RobotProfile) -> RobotArticulationCfg:
    """Build the Isaac Lab articulation config for a robot from its profile.

    Actuator groups map to :class:`ImplicitActuatorCfg` (``pd_mode="implicit"``,
    the mode used by every launch robot trained through this base). Explicit-PD
    robots need an IdealPDActuator mapping which lands together with the first
    explicit-PD robot -- constructing them silently as implicit would flip the
    sim-to-real behavior, so we refuse instead.
    """
    if profile.pd_mode == "implicit":
        pass  # ImplicitActuatorCfg below
    elif profile.pd_mode == "explicit":
        raise NotImplementedError(
            f"RobotProfile {profile.vendor}/{profile.model}: pd_mode='explicit' is not "
            "supported by build_articulation_cfg yet (lands with the first explicit-PD "
            "robot). Refusing to guess an actuator model."
        )
    else:
        # RobotProfile.__post_init__ already validates, but consumers must not
        # rely on that silently (project rule: validate pd_mode at every consumer).
        raise ValueError(f"Unknown pd_mode {profile.pd_mode!r}; expected 'implicit' or 'explicit'.")

    # Spawn source: USD when provided, otherwise the URDF (upstream G1 flow).
    if profile.usd is not None:
        spawn = RobotUsdFileCfg(usd_path=str(profile.asset_path("usd")))
    else:
        spawn = RobotUrdfFileCfg(asset_path=str(profile.asset_path("urdf")))
    spawn.articulation_props.enabled_self_collisions = profile.self_collisions

    actuators = {}
    for group_name, group in profile.actuator_groups.items():
        actuators[group_name] = ImplicitActuatorCfg(
            joint_names_expr=list(group.joint_names_expr),
            effort_limit_sim=group.effort_limit,
            velocity_limit_sim=group.velocity_limit,
            stiffness=dict(group.stiffness) if isinstance(group.stiffness, dict) else group.stiffness,
            damping=dict(group.damping) if isinstance(group.damping, dict) else group.damping,
            armature=group.armature,
        )

    return RobotArticulationCfg(
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, profile.spawn_height_m),
            joint_pos=dict(profile.default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
        joint_sdk_names=list(profile.joint_sdk_names) if profile.joint_sdk_names is not None else None,
    )
