import os
import site

# Ensure Torch DLLs are loaded correctly (Windows specific safeguard)
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
    # CRITICAL FIX: Add .unwrapped to bypass the SB3 Monitor and access the raw env attributes
    router = OldRouter(env.unwrapped)
    return router.getReward(b_path)

def EvaluatePolicy(model, env, n_eval_episodes = 100, b_path = False):
    episode_rewards = []
    legacy_rewards = []
    n_steps = 0
    for i in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
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
        # LOGGING UPDATE: verbose=1 turns on SB3's detailed, real-time terminal output
        model = PPO(MyCnnPolicy, env, n_steps = policy_steps, verbose=1)
        
    agent_rewards = []
    old_rewards = []
    episodes = []
    
    print("\n" + "="*60)
    print("### INITIALIZING NEW EXPERIMENT ###")
    print("Algorithm: PPO")
    print("Detailed Logging: ENABLED (verbose=1)")
    print("CNN Extractor Architecture Loaded:")
    print(model.policy.features_extractor)
    print("="*60 + "\n")
    
    for i in range(num_iterations + 1):
        print(f"\n--- Starting Iteration {i}/{num_iterations} (Training for {num_steps} steps) ---")
        
        # 1. Train the model
        model.learn(total_timesteps = num_steps)
        
        # 2. Evaluate the model
        mean_reward, n_steps, legacy_reward = EvaluatePolicy(model, model.get_env(), n_eval_episodes = 50, b_path = b_path)
        
        # 3. Calculate current dynamic learning rate
        progress_remaining = 1.0 - (i / num_iterations)
        current_lr = model.lr_schedule(progress_remaining) if hasattr(model, 'lr_schedule') else model.learning_rate
        
        # 4. End of iteration summary
        print(f"\n[Summary] Iteration {i} Completed:")
        print(f"      -> Agent Mean Reward : {mean_reward:.4f}")
        print(f"      -> Baseline Reward   : {legacy_reward:.4f}")
        print(f"      -> Current LR        : {current_lr:.6f}")
        print("-" * 60)
        
        agent_rewards.append(mean_reward)
        old_rewards.append(legacy_reward)
        episodes.append(i)
        
    agent_rewards = agent_rewards[-num_iterations:]
    old_rewards = old_rewards[-num_iterations:]
    episodes = episodes[:num_iterations]
    
    # Return the trained model so we can extract its weights in expSeveralRuns
    return agent_rewards, old_rewards, episodes, model

def showIsGPU():
    if torch.cuda.is_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")

def plotAgentPerformance(a_rewards, o_rewards, size, env_info, b_path = False):
    a_rewards = np.array(a_rewards)
    o_rewards = np.array(o_rewards)
    a_line = np.average(a_rewards, axis = 0)
    o_line = np.average(o_rewards, axis = 0)
    a_max = np.max(a_rewards, axis = 0)
    a_min = np.min(a_rewards, axis = 0)
    o_max = np.max(o_rewards, axis = 0)
    o_min = np.min(o_rewards, axis = 0)
    episodes = list(range(len(a_max)))
    
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 20})
        plt.figure()
        plt.fill_between(episodes, a_max, a_min, facecolor = 'red', alpha = 0.3)
        plt.fill_between(episodes, o_max, o_min, facecolor = 'blue', alpha = 0.3)
        plt.plot(episodes, a_line, 'r-', label = 'Agent')
        plt.plot(episodes, o_line, 'b-', label = 'Baseline')
        
        if b_path:
            leg = plt.legend(loc = 'upper left', shadow = True, fancybox = True)
        else:
            leg = plt.legend(loc = 'lower right', shadow = True, fancybox = True)
        leg.get_frame().set_alpha(0.5)
        plt.title("DMFB " + size)
        plt.xlabel('Training Epochs')
        
        if b_path:
            plt.ylabel('Number of Cycles')
        else:
            plt.ylabel('Score')
            
        plt.tight_layout()
        
        os.makedirs('log', exist_ok=True)
        plt.savefig('log/' + size + env_info + '.png')

def expSeveralRuns(args, n_e, n_s, n_repeat):
    size = str(args['w']) + 'x' + str(args['l'])
    env_info = '_m' + str(args['n_modules'])
    
    env = make_vec_env(DMFBEnv, n_envs = n_e, env_kwargs = args)
    showIsGPU()
    
    a_rewards = []
    o_rewards = []
    final_model = None # Variable to hold our trained model
    
    for i in range(n_repeat):
        print(f"\n>>> Starting Repeat Run {i+1}/{n_repeat} <<<")
        # Catch the returned model here
        a_r, o_r, episodes, trained_model = runAnExperiment(env, num_iterations = 2, num_steps = 20, policy_steps = n_s)
        a_rewards.append(a_r)
        o_rewards.append(o_r)
        final_model = trained_model
        
    plotAgentPerformance(a_rewards, o_rewards, size, env_info)
    
    # ==========================================
    # WEIGHT EXTRACTION
    # ==========================================
    print("\n" + "="*60)
    print("### FINAL MODEL WEIGHTS ###")
    
    print("\nExtracting CNN Weights...")
    # Target the features extractor (the Table1CNN) inside the PPO policy
    cnn_state_dict = final_model.policy.features_extractor.state_dict()
    
    for layer_name, weight_tensor in cnn_state_dict.items():
        print(f"\n-> Layer: {layer_name} | Shape: {weight_tensor.shape}")
        # Convert the PyTorch tensor to a numpy array and print it
        print(weight_tensor.cpu().numpy())
        
    print("="*60 + "\n")

sizes = [10]
for s in sizes:
    args = {'w': s, 'l': s, 'n_modules': 0, 'b_degrade': False, 'per_degrade': 0.0}
    expSeveralRuns(args, n_e = 8, n_s = 64, n_repeat = 3)
print('### Finished train.py successfully ###')