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

def plotEvaluation(std_cycles, wear_cycles, base_cycles, size):
    std_cycles = np.array(std_cycles)
    wear_cycles = np.array(wear_cycles)
    base_cycles = np.array(base_cycles)
    
    episodes = list(range(len(std_cycles)))
    
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 6))
        
        plt.plot(episodes, std_cycles, 'r-', label='Standard RL Agent', linewidth=2)
        plt.plot(episodes, wear_cycles, 'g-', label='Wear-Leveling Agent (\u03BB=0.0015)', linewidth=2)
        plt.plot(episodes, base_cycles, 'b--', label='Static Baseline', linewidth=2)
        
        plt.legend(loc='upper right', shadow=True, fancybox=True)
        plt.title(f"DMFB {size} Parallel Degradation Evaluation (50%)")
        plt.xlabel('Adaptation Epochs')
        plt.ylabel('Number of Cycles') 
        
        plt.tight_layout()
        os.makedirs('log', exist_ok=True)
        plt.savefig(f'log/parallel_eval_{size}.png')
        print(f"\n[Success] Plot saved to log/parallel_eval_{size}.png", flush=True)

if __name__ == '__main__':
    size_str = "10x10"
    saved_model_name = "ppo_dmfb_model_10x10" 
    
    num_iterations = 40
    num_steps_per_epoch = 20000
    n_eval_episodes = 50

    print("\n" + "="*80, flush=True)
    print("### INITIALIZING PARALLEL EVALUATION ENVIRONMENTS ###", flush=True)
    
    # 1. Ensure Identical Chip Layouts (50% Degradation map)
    # By using the exact same seed before creating the environments, 
    # the 50% degraded electrodes will be in the exact same physical locations.
    np.random.seed(42)
    args_std = {'w': 10, 'l': 10, 'n_modules': 0, 'b_degrade': True, 'per_degrade': 0.5, 'use_wear_leveling': False, 'b_random': False}
    env_std = make_vec_env(DMFBEnv, n_envs=8, env_kwargs=args_std)
    
    np.random.seed(42)
    args_wear = {'w': 10, 'l': 10, 'n_modules': 0, 'b_degrade': True, 'per_degrade': 0.5, 'use_wear_leveling': True, 'b_random': False}
    env_wear = make_vec_env(DMFBEnv, n_envs=8, env_kwargs=args_wear)

    # Load the identical healthy starting brains
    model_std = PPO.load(saved_model_name, env=env_std)
    model_wear = PPO.load(saved_model_name, env=env_wear)

    history_std = []
    history_wear = []
    history_base = []

    print("="*80 + "\n", flush=True)

    # Parallel Execution Loop
    for epoch in range(num_iterations):
        print(f"--- Adaptation Epoch {epoch+1}/{num_iterations} ---", flush=True)
        
        # 1. Both agents learn on their respective (increasingly degraded) boards
        model_std.learn(total_timesteps=num_steps_per_epoch,log_interval=500)
        model_wear.learn(total_timesteps=num_steps_per_epoch,log_interval=500)
        
        # 2. Testing Phase
        epoch_std_cycles = []
        epoch_wear_cycles = []
        epoch_base_cycles = []

        # Generate 50 identical tasks for this epoch
        random.seed(100 + epoch) # Ensures tasks change per epoch, but are identical for all agents
        
        for task in range(n_eval_episodes):
            start_pos = (random.randrange(0, 10), random.randrange(0, 10))
            end_pos = (random.randrange(0, 10), random.randrange(0, 10))
            while end_pos == start_pos:
                end_pos = (random.randrange(0, 10), random.randrange(0, 10))

            # A. Test Standard Agent
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

            # B. Test Wear-Leveling Agent
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

            # C. Test Static Baseline (Using std board to calculate static math)
            legacy_r = legacyReward(env_std.envs[0], b_path=True)
            epoch_base_cycles.append(legacy_r)

        # 3. Log Averages for the Epoch
        avg_std = np.mean(epoch_std_cycles)
        avg_wear = np.mean(epoch_wear_cycles)
        avg_base = np.mean(epoch_base_cycles)

        print(f"    -> Static Baseline Cycles: {avg_base:.2f}", flush=True)
        print(f"    -> Standard Agent Cycles : {avg_std:.2f}", flush=True)
        print(f"    -> Wear-Leveling Cycles  : {avg_wear:.2f}\n", flush=True)

        history_std.append(avg_std)
        history_wear.append(avg_wear)
        history_base.append(avg_base)

    # Final Output
    print("\n" + "="*80, flush=True)
    print("### FINAL ADAPTATION CYCLES PER EPOCH (50% DEGRADATION) ###", flush=True)
    print(f"Static Baseline Cycles : {[round(num, 2) for num in history_base]}", flush=True)
    print(f"Standard Agent Cycles  : {[round(num, 2) for num in history_std]}", flush=True)
    print(f"Wear-Leveling Cycles   : {[round(num, 2) for num in history_wear]}", flush=True)
    print("="*80 + "\n", flush=True)

    plotEvaluation(history_std, history_wear, history_base, size_str)
