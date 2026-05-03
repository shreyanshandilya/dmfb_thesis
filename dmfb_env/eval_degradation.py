import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch 

from utils import OldRouter
from envs.dmfb import *
from my_net import MyCnnPolicy, Table1CNN

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

def legacyReward(env, b_path=False):
    router = OldRouter(env.unwrapped)
    return router.getReward(b_path)

def plotEvaluation(std_data, wear_data, base_data, size):
    # Unpack the mean, min, and max tuples
    std_mean, std_min, std_max = zip(*std_data)
    wear_mean, wear_min, wear_max = zip(*wear_data)
    base_mean, base_min, base_max = zip(*base_data)
    
    episodes = list(range(len(std_mean)))
    
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 6))
        
        # Plot the average lines
        plt.plot(episodes, std_mean, 'r-', label='Standard RL Agent', linewidth=2)
        plt.plot(episodes, wear_mean, 'g-', label='Wear-Leveling Agent (exp_cap)', linewidth=2)
        plt.plot(episodes, base_mean, 'b--', label='Static Baseline', linewidth=2)
        
        # Plot the shaded regions representing best (min) and worst (max) performance
        plt.fill_between(episodes, std_min, std_max, color='red', alpha=0.2)
        plt.fill_between(episodes, wear_min, wear_max, color='green', alpha=0.2)
        plt.fill_between(episodes, base_min, base_max, color='blue', alpha=0.2)
        
        plt.legend(loc='upper right', shadow=True, fancybox=True)
        plt.title(f"DMFB {size} Parallel Degradation Evaluation (50%)")
        plt.xlabel('Adaptation Epochs')
        plt.ylabel('Number of Cycles') 
        
        plt.tight_layout()
        os.makedirs('log', exist_ok=True)
        plt.savefig(f'log/parallel_eval_shaded_{size}.png')
        print(f"\n[Success] Shaded Plot saved to log/parallel_eval_shaded_{size}.png", flush=True)

if __name__ == '__main__':
    size_str = "10x10"
    saved_model_name = "ppo_dmfb_model_10x10" 
    
    num_iterations = 100
    num_steps_per_epoch = 20000
    n_eval_episodes = 5 # Run 5 tasks per epoch for variance shading

    print("\n" + "="*80, flush=True)
    print("### INITIALIZING PARALLEL EVALUATION ENVIRONMENTS ###", flush=True)
    
    np.random.seed(42)
    args_std = {'w': 10, 'l': 10, 'n_modules': 0, 'b_degrade': True, 'per_degrade': 0.5, 'use_wear_leveling': False, 'b_random': False}
    env_std = make_vec_env(DMFBEnv, n_envs=8, env_kwargs=args_std)
    
    np.random.seed(42)
    args_wear = {'w': 10, 'l': 10, 'n_modules': 0, 'b_degrade': True, 'per_degrade': 0.5, 'use_wear_leveling': True, 'b_random': False}
    env_wear = make_vec_env(DMFBEnv, n_envs=8, env_kwargs=args_wear)

    # Set verbose=0 to hide SB3 boxes
    model_std = PPO.load(saved_model_name, env=env_std, verbose=0)
    model_wear = PPO.load(saved_model_name, env=env_wear, verbose=0)

    history_std = []
    history_wear = []
    history_base = []

    print("="*80 + "\n", flush=True)

    for epoch in range(num_iterations):
        print(f"--- Adaptation Epoch {epoch+1}/{num_iterations} ---", flush=True)
        
        # Adaptation Phase (Learn)
        model_std.learn(total_timesteps=num_steps_per_epoch)
        model_wear.learn(total_timesteps=num_steps_per_epoch)
        
        epoch_std_cycles = []
        epoch_wear_cycles = []
        epoch_base_cycles = []

        # 5 Exact same random tasks to calculate variance
        random.seed(100 + epoch) 
        
        for task in range(n_eval_episodes):
            start_pos = (random.randrange(0, 10), random.randrange(0, 10))
            end_pos = (random.randrange(0, 10), random.randrange(0, 10))
            while end_pos == start_pos:
                end_pos = (random.randrange(0, 10), random.randrange(0, 10))

            # Test Standard
            env_std.envs[0].unwrapped.injected_start = start_pos
            env_std.envs[0].unwrapped.injected_end = end_pos
            obs_std = env_std.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model_std.predict(obs_std)
                obs_std, _, done, _ = env_std.step(action)
                done = done[0]
                steps += 1
            epoch_std_cycles.append(steps)

            # Test Wear-Leveling
            env_wear.envs[0].unwrapped.injected_start = start_pos
            env_wear.envs[0].unwrapped.injected_end = end_pos
            obs_wear = env_wear.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model_wear.predict(obs_wear)
                obs_wear, _, done, _ = env_wear.step(action)
                done = done[0]
                steps += 1
            epoch_wear_cycles.append(steps)

            # Test Baseline
            legacy_r = legacyReward(env_std.envs[0], b_path=True)
            epoch_base_cycles.append(legacy_r)

        # Record (Mean, Min, Max) tuples for the plot
        history_std.append((np.mean(epoch_std_cycles), np.min(epoch_std_cycles), np.max(epoch_std_cycles)))
        history_wear.append((np.mean(epoch_wear_cycles), np.min(epoch_wear_cycles), np.max(epoch_wear_cycles)))
        history_base.append((np.mean(epoch_base_cycles), np.min(epoch_base_cycles), np.max(epoch_base_cycles)))

        print(f"    -> Static Baseline Cycles: {history_base[-1][0]:.2f}", flush=True)
        print(f"    -> Standard Agent Cycles : {history_std[-1][0]:.2f}", flush=True)
        print(f"    -> Wear-Leveling Cycles  : {history_wear[-1][0]:.2f}\n", flush=True)

    plotEvaluation(history_std, history_wear, history_base, size_str)
    # ---> ADD THIS BLOCK AT THE VERY END OF THE SCRIPT <---
    print("\n" + "-"*80, flush=True)
    print("### FINAL WEAR-LEVELING VARIANCE ANALYSIS ###", flush=True)
    
    # Extract the permanent usage matrices from the environments
    std_usage_matrix = env_std.envs[0].unwrapped.m_usage
    wear_usage_matrix = env_wear.envs[0].unwrapped.m_usage
    
    # Calculate Variance (Lower variance = better wear leveling)
    std_variance = np.var(std_usage_matrix[env_std.envs[0].unwrapped.m_degrade < 1.0])
    wear_variance = np.var(wear_usage_matrix[env_wear.envs[0].unwrapped.m_degrade < 1.0])
    
    print(f"Standard Agent Usage Variance: {std_variance:.2f}", flush=True)
    print(f"Wear-Leveling Usage Variance : {wear_variance:.2f}", flush=True)
    print("-" * 80 + "\n", flush=True)
