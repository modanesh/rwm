"""
Standalone script to visualize ANYmal D policy with custom states and IMAGINATION.

This script demonstrates how to:
1. Initialize the ANYmal D environment using the visualization configuration.
2. Set custom joint positions, velocities, and root state.
3. Visualize the "imagination" (world model prediction) by resetting the imagination environments
   to the predicted state at each step.

Note: This script is standalone and copies necessary classes/functions from the local `mbrl` package.
"""

import argparse
import torch
import math
from typing import Any, Tuple

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Visualize ANYmal D policy with custom states and imagination.")
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
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_apply
from tensordict import TensorDict

# --- Copied from mbrl.mbrl.envs.mdp.events ---

def reset_root_state_to_specified(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    velocities: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_to_specified(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_root_velocity_to_specified(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocities: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# --- Mocks for System Dynamics and Normalizers ---

class DummyNormalizer:
    def __init__(self):
        pass
    def forward(self, x):
        return x
    def inverse(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

class DummySystemDynamics:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.architecture_config = {"type": "mlp"} # Mock config

    def forward(self, state_history, action_history):
        """
        Mock forward pass. Returns next state.

        state_history: (num_envs, horizon, state_dim)
        action_history: (num_envs, horizon, action_dim)
        """
        # Simple dynamics: current state + small noise/drift to visualize change
        # Assuming state_history[:, -1] is current state.
        current_state = state_history[:, -1]

        # We need to return full set of outputs expected by _update_imagination_envs
        # imagination_states, *_ = self.system_dynamics.forward(...)

        # Let's just return current state + 0.01 to simulate some movement
        next_state = current_state.clone()
        # Create some fake movement in position/velocity parts if possible
        # but since we don't know exact layout here without parsing, we just add small value
        next_state += 0.001

        # Return tuple as expected
        return next_state, None, None, None, None, None


# --- Copied from mbrl.mbrl.envs.manager_based_mbrl_env ---

class ManagerBasedMBRLEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_term_names = self.reward_manager.active_terms
        self.reward_term_names.append("uncertainty")
        self._init_additional_attributes()
        # assigned in runner
        self.num_imagination_envs = None # type: int
        self.num_imagination_steps = None # type: int
        self.imagination_state_normalizer = None # type: Any
        self.imagination_action_normalizer = None # type: Any
        self.system_dynamics = None # type: Any
        self.uncertainty_penalty_weight = None # type: float
        # termination flags
        self.termination_flags = None # type: torch.Tensor | None

    def prepare_imagination(self):
        # We don't implement full preparation here as we don't use the full MBRL loop
        # But we might need attributes
        pass

    def _init_additional_attributes(self):
        raise NotImplementedError

    def _prepare_additional_imagination_attributes(self):
        raise NotImplementedError

    def _parse_imagination_states(self, imagination_states_denormalized):
        raise NotImplementedError

    def _parse_extensions(self, extensions):
        raise NotImplementedError

    def _parse_contacts(self, contacts):
        raise NotImplementedError

    def _parse_terminations(self, terminations):
        raise NotImplementedError

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        raise NotImplementedError


# --- Copied from mbrl.mbrl.envs.manager_based_visualize_env ---

class ManagerBasedVisualizeEnv(ManagerBasedMBRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.env_ids_real = torch.arange(0, self.num_envs, 2, device=self.device)
        self.env_ids_imagination = torch.arange(1, self.num_envs, 2, device=self.device)

    def init_imagination_history(self, history_horizon):
        self.imagination_state_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_state"][0], device=self.device)
        self.imagination_action_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_action"][0], device=self.device)

    def _sync_imagination_history(self, env_ids_real):
        self.imagination_state_history[env_ids_real // 2] = 0.0
        self.imagination_action_history[env_ids_real // 2] = 0.0
        self.imagination_state_history[env_ids_real // 2, -1] = self.imagination_state_normalizer(self.observation_manager.compute()["system_state"])[env_ids_real]

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
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
            # write imagination states
            if i == self.cfg.decimation - 1:
                self._update_imagination_envs(action)
            # simulate
            self.sim.step(render=False)

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            uniques, counts = torch.cat([reset_env_ids, self.env_ids_real]).unique(return_counts=True)
            env_ids_real = uniques[counts > 1]
            env_ids_imagination = env_ids_real + 1
            env_ids = torch.vstack([env_ids_real, env_ids_imagination]).T.flatten()
            if len(env_ids) > 0:
                self._reset_idx(env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute()
        if len(reset_env_ids) > 0 and len(env_ids_real) > 0:
            self._sync_imagination_history(env_ids_real)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _update_imagination_envs(self, action):
        self.num_imagination_envs = len(self.env_ids_imagination)
        rollout_action = action[self.env_ids_imagination]
        self.imagination_action_history = torch.cat([self.imagination_action_history[:, 1:].clone(), self.imagination_action_normalizer(rollout_action).unsqueeze(1)], dim=1)

        if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm", "mlp"]: # Added mlp for dummy
             # Original code handled rnn/rssm specifically. For MLP usually simple concatenation.
             # The original code did:
             # if type in ["rnn", "rssm"]: ...
             # We will adapt to what we see. Original:
             if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]:
                self.imagination_state_history = self.imagination_state_history[:, -1].unsqueeze(1)
                self.imagination_action_history = self.imagination_action_history[:, -1].unsqueeze(1)

        imagination_states, *_ = self.system_dynamics.forward(self.imagination_state_history, self.imagination_action_history)

        # NOTE: DummySystemDynamics returns (num_envs, state_dim).
        # But _update_imagination_envs expects imagination_states to be valid for normalizer.inverse

        imagination_states_denormalized = self.imagination_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        self._reset_imagination_sim(parsed_imagination_states)

        # Update history
        # If we reduced to 1 step for RNN, we might need to handle history differently for next step
        # But for visualization, we just append.
        if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]:
             # If it was reduced, we can't cat easily if we don't restore.
             # But here we just want it to work.
             pass
        else:
             self.imagination_state_history = torch.cat([self.imagination_state_history[:, 1:].clone(), imagination_states.unsqueeze(1)], dim=1)


    def _reset_imagination_sim(self, parsed_imagination_states):
        raise NotImplementedError


# --- Copied from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_mbrl_env ---

class ANYmalDManagerBasedMBRLEnv(ManagerBasedMBRLEnv):

    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0]
        self.base_velocity = None

    def _prepare_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states

    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {
        }
        return parsed_extensions

    def _parse_contacts(self, contacts):
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None

        parsed_contacts = {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }
        return parsed_contacts

    def _parse_terminations(self, terminations):
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().int() if terminations is not None else None
        return parsed_terminations

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        pass # Not needed for visualization if we don't care about rewards

# --- Copied from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_visualize_env ---

class ANYmalDManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, ANYmalDManagerBasedMBRLEnv):

    def _reset_imagination_sim(self, parsed_imagination_states):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_pos += self.default_joint_pos
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_vel += self.default_joint_vel

        root_quat_w = self.scene["robot"].data.root_quat_w[self.env_ids_imagination]

        base_lin_vel_w = quat_apply(root_quat_w, base_lin_vel)
        base_ang_vel_w = quat_apply(root_quat_w, base_ang_vel)

        velocities = torch.cat([base_lin_vel_w, base_ang_vel_w], dim=1)

        reset_joints_to_specified(self, self.env_ids_imagination, joint_pos, joint_vel)
        reset_root_velocity_to_specified(self, self.env_ids_imagination, velocities)


# --- Configuration (Same as before) ---

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as isaac_mdp

@configclass
class StandaloneObservationsCfg(ObservationsCfg):
    @configclass
    class SystemStateCfg(ObsGroup):
        # Must match _parse_imagination_states expectations in order:
        # base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, joint_torque
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=isaac_mdp.joint_effort)

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
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.scene.num_envs = args.num_envs
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        self.observations = StandaloneObservationsCfg()


def main():
    # 1. Configure the environment
    env_cfg = StandaloneAnymalDFlatEnvCfg()

    # 2. Create the environment
    env = ANYmalDManagerBasedVisualizeEnv(cfg=env_cfg, render_mode="rgb_array")

    # 3. Setup Mock Dynamics and Normalizers
    # We need to know state dimension from observations.
    # We can run one reset to get the observation shape.
    obs, _ = env.reset()

    print("[INFO] Setting up imagination...")
    # Initialize imagination history
    history_horizon = 2
    env.init_imagination_history(history_horizon)

    # Attach dummy components
    env.imagination_state_normalizer = DummyNormalizer()
    env.imagination_action_normalizer = DummyNormalizer()

    # Check observation sizes
    state_dim = env.observation_manager.group_obs_dim["system_state"][0]
    action_dim = env.observation_manager.group_obs_dim["system_action"][0]

    env.system_dynamics = DummySystemDynamics(state_dim, action_dim)

    # 4. Set custom state (Real envs mostly, but imagination starts from here)
    print("[INFO] Setting custom state...")
    num_envs = env.num_envs

    # Joint positions
    default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
    if default_joint_pos.shape[0] == 1:
        default_joint_pos = default_joint_pos.repeat(num_envs, 1)

    custom_joint_pos = default_joint_pos.clone()
    custom_joint_vel = torch.zeros_like(custom_joint_pos)

    # Root state
    custom_root_pos = torch.zeros((num_envs, 3), device=env.device)
    custom_root_pos[:, 0] = 0.0
    custom_root_pos[:, 1] = 0.0
    custom_root_pos[:, 2] = 0.60

    custom_root_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)
    custom_root_vel = torch.zeros((num_envs, 6), device=env.device)
    custom_root_vel[:, 0] = 0.5

    env_ids = torch.arange(num_envs, device=env.device)

    reset_joints_to_specified(env, env_ids, custom_joint_pos, custom_joint_vel)
    reset_root_state_to_specified(env, env_ids, custom_root_pos, custom_root_rot, custom_root_vel)

    # Sync history so imagination starts correctly
    env._sync_imagination_history(env.env_ids_real)

    # 5. Simulation loop
    print("[INFO] Starting simulation loop with imagination...")
    while simulation_app.is_running():
        # Get action from policy (dummy)
        action = torch.zeros((num_envs, env.action_manager.action_dim), device=env.device)

        # Step the environment
        # This will trigger _update_imagination_envs inside step()
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
