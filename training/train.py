import torch
import numpy as np
import time
import os
import yaml
from collections import deque

# Select environment (adjust import based on structure)
# from environments.ant.ant_env import AntEnv as SelectedEnv
# Placeholder: Assume a function get_env() exists
def get_env(env_name="ant", device='cuda'):
     if env_name.lower() == "ant":
         from environments.ant.ant_env import AntEnv
         return AntEnv(device=device)
     # elif env_name.lower() == "gripper":
     #     from environments.gripper.gripper_env import GripperEnv
     #     return GripperEnv(device=device)
     else:
         raise ValueError(f"Unknown environment: {env_name}")

from models.sac import SAC
from .replay_buffer import ReplayBuffer
from utils.logger import Logger # Simple logger utility

def train(config_path="configs/ant.yaml", device='cuda'):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_config = config['environment']
    sac_config = config['sac_agent']
    train_config = config['training']

    # --- Initialization ---
    env = get_env(env_config['name'], device=device)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = SAC(obs_dim=obs_dim,
                action_dim=action_dim,
                action_space_low=action_low,
                action_space_high=action_high,
                device=device,
                **sac_config) # Pass SAC hyperparameters

    replay_buffer = ReplayBuffer(obs_dim, action_dim, train_config['buffer_capacity'], device)

    # Logging setup
    log_dir = "logs/" + env_config['name'] + "_" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # --- Training Loop ---
    start_time = time.time()
    obs = env.reset()
    episode_reward = 0
    episode_len = 0
    episode_num = 0
    recent_rewards = deque(maxlen=10) # Track recent rewards for progress

    for global_step in range(train_config['total_steps']):
        # Select action
        if global_step < train_config['start_steps']:
            action = env.action_space.sample() # Random actions initially
        else:
            action = agent.select_action(obs)

        # Step environment
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_len += 1

        # Store transition (handle episode termination properly)
        # Store actual 'done' flag, but use 'terminal' for value calculation if timeout
        terminal = done # Simple case: done means terminal
        # More robust: Check if done was due to timeout (not failure)
        # terminal = False if episode_len == env._max_episode_steps and done else done

        replay_buffer.add(obs, action, reward, next_obs, terminal)

        obs = next_obs

        # --- Agent Updates ---
        if global_step >= train_config['start_steps'] and global_step % train_config['update_freq'] == 0:
            for _ in range(train_config['update_freq']): # Perform multiple updates per env step
                if replay_buffer.size < train_config['batch_size']: break # Wait for enough samples

                batch = replay_buffer.sample(train_config['batch_size'])

                # Decide update type: RL or Physics Gradient
                use_physics_grad = (train_config['physics_grad_ratio'] > 0 and
                                    np.random.rand() < train_config['physics_grad_ratio'])
                                    # global_step % round(1.0 / train_config['physics_grad_ratio']) == 0) # Alternate fixed ratio

                if use_physics_grad:
                     # Ensure obs in batch has grad enabled if needed by physics update implementation
                     # Physics update needs access to the simulation model and state structure
                     physics_loss = agent.update_physics_gradient(batch, env.model, env.state_template)
                     logger.log_scalar('Loss/PhysicsActorLoss', physics_loss, global_step)
                     # Also perform a standard RL update? Or purely alternate? Configurable.
                     if train_config.get('interleave_rl_update', True):
                          critic_loss, actor_loss = agent.update_rl(batch)
                          logger.log_scalar('Loss/CriticLoss', critic_loss, global_step)
                          logger.log_scalar('Loss/RLActorLoss', actor_loss, global_step)
                          logger.log_scalar('Stats/Alpha', agent.alpha, global_step)

                else: # Standard RL update
                    critic_loss, actor_loss = agent.update_rl(batch)
                    logger.log_scalar('Loss/CriticLoss', critic_loss, global_step)
                    logger.log_scalar('Loss/RLActorLoss', actor_loss, global_step)
                    logger.log_scalar('Stats/Alpha', agent.alpha, global_step)


        # --- Episode End Handling ---
        if done:
            episode_num += 1
            recent_rewards.append(episode_reward)
            avg_reward = np.mean(recent_rewards)

            print(f"Step: {global_step}, Episode: {episode_num}, Length: {episode_len}, Reward: {episode_reward:.2f}, AvgReward: {avg_reward:.2f}")
            logger.log_scalar('Reward/EpisodeReward', episode_reward, global_step)
            logger.log_scalar('Reward/AvgReward_Last10', avg_reward, global_step)
            logger.log_scalar('Stats/EpisodeLength', episode_len, global_step)

            # Reset
            obs = env.reset()
            episode_reward = 0
            episode_len = 0

        # --- Logging & Checkpointing ---
        if global_step % train_config['log_interval'] == 0:
             elapsed_time = time.time() - start_time
             print(f"Step: {global_step}, Time: {elapsed_time:.2f}s")
             logger.dump(global_step)

        if global_step % train_config['checkpoint_interval'] == 0 and global_step > 0:
             agent.save(os.path.join(log_dir, f'sac_checkpoint_{global_step}.pth'))

    print("Training finished.")
    agent.save(os.path.join(log_dir, 'sac_final.pth'))

if __name__ == "__main__":
    # TODO: Add argparse for config file, device selection etc.
    train(config_path="configs/ant.yaml", device='cuda' if torch.cuda.is_available() else 'cpu')