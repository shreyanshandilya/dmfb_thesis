import os
import site

for path in site.getsitepackages() + [site.getusersitepackages()]:
    torch_lib = os.path.join(path, "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)

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

def expSeveralRuns(args, n_e, n_s, n_repeat):
    size = str(args['w']) + 'x' + str(args['l'])
    env = make_vec_env(DMFBEnv, n_envs = n_e, env_kwargs = args)
    showIsGPU()
    
    final_model = None
    
    for i in range(n_repeat):
        print(f"\n>>> Starting Run {i+1}/{n_repeat} <<<")
        a_r, o_r, episodes, trained_model = runAnExperiment(env, num_iterations = 2, num_steps = 20, policy_steps = n_s)
        final_model = trained_model
        
    # Extract Weights and Save Final Model
    print("\n" + "="*60)
    print("### FINAL MODEL EXPORT ###")
    cnn_state_dict = final_model.policy.features_extractor.state_dict()
    
    for layer_name, weight_tensor in cnn_state_dict.items():
        print(f"\n-> Layer: {layer_name} | Shape: {weight_tensor.shape}")
        print(weight_tensor.cpu().numpy())
        
    final_model.save(f"ppo_dmfb_model_{size}")
    print(f"\n[Success] Full PPO model saved to: ppo_dmfb_model_{size}.zip")
    print("="*60 + "\n")

if __name__ == '__main__':
    sizes = [10]
    for s in sizes:
        # Healthy mode: 0% degradation, NO wear-leveling
        args = {'w': s, 'l': s, 'n_modules': 0, 'b_degrade': False, 'per_degrade': 0.0, 'use_wear_leveling': False}
        expSeveralRuns(args, n_e = 8, n_s = 64, n_repeat = 1)