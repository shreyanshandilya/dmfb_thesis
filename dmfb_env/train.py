import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch 

from utils import OldRouter
from envs.dmfb import *
from my_net import MyCnnPolicy 

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

def legacyReward(env, b_path = False):
    router = OldRouter(env.unwrapped)
    return router.getReward(b_path)

def EvaluatePolicy(model, env, n_eval_episodes = 100, b_path = False):
    episode_rewards = []
    legacy_rewards = []
    n_steps = 0
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        this_loop_steps = 0
        while not done:
            action, state = model.predict(obs)
            obs, reward, done, _info = env.step(action)
            reward = reward[0]
            done = done[0]
            episode_reward += reward
            n_steps += 1
            this_loop_steps += 1
            
            legacy_r = legacyReward(env.envs[0], b_path)
            
        if b_path:
            episode_rewards.append(this_loop_steps)
        else:
            episode_rewards.append(episode_reward)
        legacy_rewards.append(legacy_r)
        
    mean_reward = np.mean(episode_rewards)
    mean_legacy = np.mean(legacy_rewards)
    return mean_reward, n_steps, mean_legacy

def runAnExperiment(env, model = None, num_iterations = 10, num_steps = 20000, policy_steps = 128, b_path = False):
    if model is None:
        model = PPO(MyCnnPolicy, env, n_steps = policy_steps, verbose=1)
        
    agent_rewards = []
    old_rewards = []
    episodes = []
    
    print("\n" + "="*60)
    print("### INITIALIZING HEALTHY TRAINING (PHASE 1) ###")
    print("="*60 + "\n")
    
    for i in range(num_iterations):
        print(f"\n--- Starting Iteration {i+1}/{num_iterations} ---")
        model.learn(total_timesteps = num_steps)
        mean_reward, n_steps, legacy_reward = EvaluatePolicy(model, model.get_env(), n_eval_episodes = 50, b_path = b_path)
        
        agent_rewards.append(mean_reward)
        old_rewards.append(legacy_reward)
        episodes.append(i)
        
    return agent_rewards, old_rewards, episodes, model

def showIsGPU():
    if torch.cuda.is_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")
        
def plotAgentPerformance(a_rewards, o_rewards, size):
    # Assuming n_repeat=1, extract the first set of runs
    a_line = np.array(a_rewards[0])
    o_line = np.array(o_rewards[0])
    episodes = list(range(len(a_line)))
    
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, a_line, 'r-', label = 'Standard RL Agent', linewidth=2)
        plt.plot(episodes, o_line, 'b-', label = 'Static Baseline', linewidth=2)
        
        plt.legend(loc = 'lower right', shadow = True, fancybox = True)
        plt.title(f"DMFB {size} Healthy Training")
        plt.xlabel('Training Epochs')
        plt.ylabel('Score')
            
        plt.tight_layout()
        os.makedirs('log', exist_ok=True)
        plt.savefig(f'log/healthy_training_{size}.png')
        print(f"\n[Success] Training curve saved to log/healthy_training_{size}.png")

def expSeveralRuns(args, n_e, n_s, n_repeat):
    size = str(args['w']) + 'x' + str(args['l'])
    env = make_vec_env(DMFBEnv, n_envs = n_e, env_kwargs = args)
    showIsGPU()
    
    a_rewards = []
    o_rewards = []
    final_model = None
    
    for i in range(n_repeat):
        print(f"\n>>> Starting Run {i+1}/{n_repeat} <<<")
        # Ensure num_iterations is set to 10
        a_r, o_r, episodes, trained_model = runAnExperiment(env, num_iterations = 10, num_steps = 20000, policy_steps = n_s)
        a_rewards.append(a_r)
        o_rewards.append(o_r)
        final_model = trained_model
        
    # --- 1. Generate the Graph ---
    plotAgentPerformance(a_rewards, o_rewards, size)
    
    # --- 2. Print Full Arrays to Terminal Logs ---
    a_line = np.array(a_rewards[0])
    o_line = np.array(o_rewards[0])
    
    print("\n" + "="*60)
    print("### FULL EPOCH REWARDS HISTORY ###")
    print(f"Agent Rewards (All {len(a_line)} Epochs): \n{a_line.tolist()}")
    print(f"\nBaseline Rewards (All {len(o_line)} Epochs): \n{o_line.tolist()}")
        
    # --- 3. Extract Weights and Save Final Model ---
    print("\n" + "="*60)
    print("### FINAL MODEL EXPORT ###")
    final_model.save(f"ppo_dmfb_model_{size}")
    print(f"\n[Success] Full PPO model saved to: ppo_dmfb_model_{size}.zip")
    print("="*60 + "\n")

if __name__ == '__main__':
    sizes = [10]
    for s in sizes:
        # Healthy mode: 0% degradation, NO wear-leveling
        args = {'w': s, 'l': s, 'n_modules': 0, 'b_degrade': False, 'per_degrade': 0.0, 'use_wear_leveling': False}
        expSeveralRuns(args, n_e = 8, n_s = 64, n_repeat = 1)
