"""
Standalone script to visualize ANYmal D policy with custom states.

This script demonstrates how to:
1. Initialize the ANYmal D environment using the visualization configuration.
2. Set custom joint positions, velocities, and root state.
3. Run a simulation loop to visualize the robot's behavior (policy execution).
"""

import argparse
import torch
import math

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
from isaaclab.envs import ManagerBasedRLEnvCfg
from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_visualize_env import ANYmalDManagerBasedVisualizeEnv
from mbrl.mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg_VISUALIZE
from mbrl.mbrl.envs.mdp.events import reset_joints_to_specified, reset_root_state_to_specified

# Define a custom environment class that skips imagination updates if no model is present
# This allows running the visualization without the full MBRL stack if desired.
class StandaloneANYmalDEnv(ANYmalDManagerBasedVisualizeEnv):
    """
    Subclass of ANYmalDManagerBasedVisualizeEnv that safely handles the absence of a system dynamics model.
    If 'system_dynamics' is missing or None, it skips the imagination update step, allowing the script
    to run with just the physics simulation.
    """
    def _update_imagination_envs(self, action):
        # If system_dynamics is not set or we want to skip imagination for this standalone script:
        if not hasattr(self, "system_dynamics") or self.system_dynamics is None:
            return
        super()._update_imagination_envs(action)

def main():
    # 1. Configure the environment
    env_cfg = AnymalDFlatEnvCfg_VISUALIZE()
    env_cfg.scene.num_envs = args.num_envs

    # 2. Create the environment
    # We use our subclass to avoid crashes if world model is missing
    env = StandaloneANYmalDEnv(cfg=env_cfg, render_mode="rgb_array")

    # 3. Reset the environment
    obs, _ = env.reset()

    # 4. Set custom state
    # Example: Slightly crouched, some forward velocity
    print("[INFO] Setting custom state...")

    num_envs = env.num_envs

    # Joint positions
    # Get default positions
    default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
    # Apply to all envs (expanding the single default row)
    if default_joint_pos.shape[0] == 1:
        default_joint_pos = default_joint_pos.repeat(num_envs, 1)

    custom_joint_pos = default_joint_pos.clone()
    # Modify: e.g., crouch slightly (this depends on joint mapping, just an example)
    # custom_joint_pos[:, :] += 0.0 # Modify as needed

    custom_joint_vel = torch.zeros_like(custom_joint_pos)

    # Root state
    # Position
    custom_root_pos = torch.zeros((num_envs, 3), device=env.device)
    custom_root_pos[:, 0] = 0.0 # X
    custom_root_pos[:, 1] = 0.0 # Y
    custom_root_pos[:, 2] = 0.60 # Z (Height)

    # Orientation (Quat: w, x, y, z)
    custom_root_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    # Velocity (Linear + Angular)
    custom_root_vel = torch.zeros((num_envs, 6), device=env.device)
    custom_root_vel[:, 0] = 0.5 # Forward linear velocity 0.5 m/s

    env_ids = torch.arange(num_envs, device=env.device)

    # Apply states
    reset_joints_to_specified(env, env_ids, custom_joint_pos, custom_joint_vel)
    reset_root_state_to_specified(env, env_ids, custom_root_pos, custom_root_rot, custom_root_vel)

    # 5. Simulation loop
    print("[INFO] Starting simulation loop...")
    while simulation_app.is_running():
        # Get action from policy
        # Replace this with actual policy inference: action = policy(obs)
        # Here we use zero actions (or random) for demonstration
        action = torch.zeros((num_envs, env.action_manager.action_dim), device=env.device)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Note: rendering is handled inside env.step() based on configuration

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
