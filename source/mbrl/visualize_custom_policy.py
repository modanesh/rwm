"""
Standalone script to visualize ANYmal D policy with custom states.

This script demonstrates how to:
1. Initialize the ANYmal D environment using the visualization configuration.
2. Set custom joint positions, velocities, and root state.
3. Run a simulation loop to visualize the robot's behavior (policy execution).

Note: This script is standalone and copies necessary classes/functions from the local `mbrl` package
to avoid direct dependency on it, as per instructions.
"""

import argparse
import torch
import math
from typing import Any

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Visualize ANYmal D policy with custom states.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
args = parser.parse_args()

# Launch the app
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

# Imports after app launch
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils

# --- Copied from mbrl.mbrl.envs.mdp.events ---

def reset_root_state_to_specified(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    velocities: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_to_specified(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

# --- Copied and Adapted from mbrl.mbrl.envs.manager_based_mbrl_env ---

class ManagerBasedMBRLEnv(ManagerBasedRLEnv):
    """
    Simplified version of ManagerBasedMBRLEnv for visualization.
    Imagination logic is stripped out or stubbed as we are in a standalone script without world model.
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # We don't need imagination attributes for pure visualization of custom policy

# --- Copied from mbrl.mbrl.envs.manager_based_visualize_env ---

from isaaclab.envs.common import VecEnvStepReturn

class ManagerBasedVisualizeEnv(ManagerBasedMBRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Original code split envs into real and imagination.
        # For this standalone script, we might just want 'real' envs, or we can keep the split if the config dictates it.
        # Assuming the user wants to control all envs, but the original class structure
        # defines env_ids_real and env_ids_imagination.
        self.env_ids_real = torch.arange(0, self.num_envs, 2, device=self.device)
        self.env_ids_imagination = torch.arange(1, self.num_envs, 2, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()

            # SKIPPING _update_imagination_envs(action) as we don't have a world model

            # simulate
            self.sim.step(render=False)

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # The original code had complex reset logic handling real vs imagination pairs.
            # We will try to preserve it to be faithful to the 'visualize env' behavior.
            uniques, counts = torch.cat([reset_env_ids, self.env_ids_real]).unique(return_counts=True)
            env_ids_real = uniques[counts > 1]
            env_ids_imagination = env_ids_real + 1
            env_ids = torch.vstack([env_ids_real, env_ids_imagination]).T.flatten()

            if len(env_ids) > 0:
                self._reset_idx(env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        self.obs_buf = self.observation_manager.compute()

        # SKIPPING _sync_imagination_history as we don't have it

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

# --- Copied from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_mbrl_env ---

class ANYmalDManagerBasedMBRLEnv(ManagerBasedMBRLEnv):
    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0]
        self.base_velocity = None

# --- Copied from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_visualize_env ---

from isaaclab.utils.math import quat_apply

class ANYmalDManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, ANYmalDManagerBasedMBRLEnv):
    # This class originally had _reset_imagination_sim but we stripped usage of imagination
    pass


# --- Configuration ---
# We need to construct the environment config without importing from mbrl.
# We will use Isaac Lab's standard AnymalD config and modify it to match AnymalDFlatEnvCfg_VISUALIZE logic where possible.

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm

# We need some functions from local mdp that were used in the config.
# Since we can't import them, we substitute them with standard Isaac Lab ones or simple mocks
# if they are just for observation definitions that we might not even strictly need for simple visual check.
# However, to be robust, we should map them to equivalent Isaac Lab functions or define them.

# Looking at flat_env_cfg.py:
# mdp.base_lin_vel -> isaac_mdp.base_lin_vel
# mdp.base_ang_vel -> isaac_mdp.base_ang_vel
# mdp.projected_gravity -> isaac_mdp.projected_gravity
# mdp.joint_pos_rel -> isaac_mdp.joint_pos_rel
# mdp.joint_vel_rel -> isaac_mdp.joint_vel_rel
# mdp.joint_effort -> isaac_mdp.joint_effort
# mdp.last_action -> isaac_mdp.last_action
# mdp.body_contact -> isaac_mdp.body_contact
# mdp.reset_root_state_uniform_visualize -> WE NEED TO DEFINE THIS or use a standard one.
#   But for this script, we are manually resetting anyway, so the automatic reset function matters less
#   unless we run for long enough to trigger automatic resets.
#   We will use standard reset_root_state_uniform from isaaclab.envs.mdp.events if available or isaac_mdp.

@configclass
class StandaloneObservationsCfg(ObservationsCfg):
    @configclass
    class SystemStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=isaac_mdp.joint_effort) # joint_effort is usually available

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        pred_actions = ObsTerm(func=isaac_mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()

@configclass
class StandaloneAnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Modify for "Flat" and "Visualize"
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Visualize specific settings
        self.scene.num_envs = args.num_envs
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Remove random pushing
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # Use our simplified observations
        # self.observations = StandaloneObservationsCfg() # Optional: strict compliance with original config
        # But simply using parent config is often enough for visualization if we don't care about specific observation contents matches.
        # Let's try to stick to the original as much as possible using standard functions.


def main():
    # 1. Configure the environment
    env_cfg = StandaloneAnymalDFlatEnvCfg()

    # 2. Create the environment
    # We use our locally defined class
    env = ANYmalDManagerBasedVisualizeEnv(cfg=env_cfg, render_mode="rgb_array")

    # 3. Reset the environment
    obs, _ = env.reset()

    # 4. Set custom state
    print("[INFO] Setting custom state...")

    num_envs = env.num_envs
    # The environment splits into real and imagination.
    # Real: 0, 2, 4... Imagination: 1, 3, 5...
    # We want to visualize on 'real' envs primarily, or both.
    # Let's set state for ALL envs.

    # Joint positions
    default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
    if default_joint_pos.shape[0] == 1:
        default_joint_pos = default_joint_pos.repeat(num_envs, 1)

    custom_joint_pos = default_joint_pos.clone()
    # Modify: e.g., crouch slightly
    # custom_joint_pos[:, :] += 0.1

    custom_joint_vel = torch.zeros_like(custom_joint_pos)

    # Root state
    custom_root_pos = torch.zeros((num_envs, 3), device=env.device)
    custom_root_pos[:, 0] = 0.0 # X
    custom_root_pos[:, 1] = 0.0 # Y
    custom_root_pos[:, 2] = 0.60 # Z (Height)

    custom_root_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    custom_root_vel = torch.zeros((num_envs, 6), device=env.device)
    custom_root_vel[:, 0] = 0.5 # Forward linear velocity

    env_ids = torch.arange(num_envs, device=env.device)

    # Apply states
    reset_joints_to_specified(env, env_ids, custom_joint_pos, custom_joint_vel)
    reset_root_state_to_specified(env, env_ids, custom_root_pos, custom_root_rot, custom_root_vel)

    # 5. Simulation loop
    print("[INFO] Starting simulation loop...")
    while simulation_app.is_running():
        # Get action from policy (dummy)
        action = torch.zeros((num_envs, env.action_manager.action_dim), device=env.device)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
