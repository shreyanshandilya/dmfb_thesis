import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from utils import OldRouter
from my_net import MyCnnPolicy
from envs.dmfb import DMFBEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def legacyReward(env, b_path=False):
    router = OldRouter(env)
    return router.getReward(b_path)

def EvaluatePolicy(model, env, n_eval_episodes=100, b_path=False):
    episode_rewards = []
    legacy_rewards = []
    n_steps = 0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        this_loop_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            
            reward = reward[0]
            done = done[0]
            
            episode_reward += reward
            n_steps += 1
            this_loop_steps += 1
            legacy_r = legacyReward(env.envs[0].unwrapped, b_path)
            
        if b_path:
            episode_rewards.append(this_loop_steps)
        else:
            episode_rewards.append(episode_reward)
        legacy_rewards.append(legacy_r)
        
    mean_reward = np.mean(episode_rewards)
    mean_legacy = np.mean(legacy_rewards)
    return mean_reward, n_steps, mean_legacy

def runAnExperiment(env, model=None, num_iterations=2, num_steps=200, policy_steps=128, b_path=False):
    if model is None:
        model = PPO(MyCnnPolicy, env, n_steps=policy_steps, verbose=0)
        
    agent_rewards = []
    old_rewards = []
    episodes = []
    
    for i in range(num_iterations + 1):
        model.learn(total_timesteps=num_steps, reset_num_timesteps=False)
        mean_reward, n_steps, legacy_reward = EvaluatePolicy(
            model, model.get_env(), n_eval_episodes=50, b_path=b_path
        )
        agent_rewards.append(mean_reward)
        old_rewards.append(legacy_reward)
        episodes.append(i)
        
    return agent_rewards[-num_iterations:], old_rewards[-num_iterations:], episodes[:num_iterations]

def find_intersection(a_rewards, o_rewards, episodes):
    for i in range(1, len(episodes)):
        if (a_rewards[i-1] < o_rewards[i-1] and a_rewards[i] >= o_rewards[i]) or \
           (a_rewards[i-1] > o_rewards[i-1] and a_rewards[i] <= o_rewards[i]):
            return episodes[i]
    return None

def plotCombinedPerformance(all_results, o_rewards, size, env_info, b_path=False):
    episodes = list(range(len(list(all_results.values())[0][0])))
    
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
        
        for idx, (lambda_val, a_rewards) in enumerate(all_results.items()):
            a_rewards = np.array(a_rewards)
            a_line = np.average(a_rewards, axis=0)
            a_max = np.max(a_rewards, axis=0)
            a_min = np.min(a_rewards, axis=0)
            
            color = colors[idx % len(colors)]
            plt.fill_between(episodes, a_max, a_min, facecolor=color, alpha=0.1)
            plt.plot(episodes, a_line, color=color, label=f'Agent (\u03bb = {lambda_val})')
            
        if o_rewards is not None:
            o_rewards = np.array(o_rewards)
            o_line = np.average(o_rewards, axis=0)
            o_max = np.max(o_rewards, axis=0)
            o_min = np.min(o_rewards, axis=0)
            plt.fill_between(episodes, o_max, o_min, facecolor='black', alpha=0.1)
            plt.plot(episodes, o_line, color='black', linestyle='--', label='Baseline')
        
        loc = 'upper left' if b_path else 'lower right'
        leg = plt.legend(loc=loc, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        
        plt.title(f"DMFB {size} Combined Lambda Curves")
        plt.xlabel('Training Epochs')
        plt.ylabel('Number of Cycles' if b_path else 'Score')
        plt.tight_layout()
        
        os.makedirs('log', exist_ok=True)
        plt.savefig(f'log/{size}{env_info}_combined_curves.png')
        plt.close()

def expSeveralRuns(args, n_e, n_s, n_repeat, lambdas):
    size = f"{args['w']}x{args['l']}"
    env_info = f"_m{args['n_modules']}"
    
    all_results = {}
    baseline_rewards_all = None
    
    for lambda_val in lambdas:
        print(f"\n========================================")
        print(f"Evaluating Lambda = {lambda_val}")
        print(f"========================================")
        
        args['penalty_lambda'] = lambda_val
        env = make_vec_env(DMFBEnv, n_envs=n_e, env_kwargs=args)
        
        a_rewards_all = []
        o_rewards_all = []
        
        for rep in range(n_repeat):
            a_r, o_r, episodes = runAnExperiment(env, num_iterations=2, num_steps=200, policy_steps=n_s)
            a_rewards_all.append(a_r)
            o_rewards_all.append(o_r)
            
        all_results[lambda_val] = a_rewards_all
        
        if lambda_val == 0:
            baseline_rewards_all = o_rewards_all
            
        a_avg = np.average(a_rewards_all, axis=0)
        o_avg = np.average(o_rewards_all, axis=0)
        
        intersection_epoch = find_intersection(a_avg, o_avg, episodes)
        
        print(f"Agent Rewards: {a_avg}")
        print(f"Baseline Rewards: {o_avg}")
        
        if intersection_epoch is not None:
            print(f"Intersection at Epoch: {intersection_epoch}")
        else:
            print("No intersection seen.")
            
        final_memory_matrix = env.envs[0].unwrapped.m_usage
        print(f"Final Memory Matrix:\n{final_memory_matrix}")
        
    plotCombinedPerformance(all_results, baseline_rewards_all, size, env_info)

if __name__ == '__main__':
    sizes = [15]
    lambdas_to_test = [0, 0.1, 0.25, 0.5, 0.75, 1] 
    
    for s in sizes:
        config = {
            'w': s, 'l': s,
            'n_modules': 0,
            'b_degrade': True,
            'per_degrade': 0.1
        }
        expSeveralRuns(config, n_e=1, n_s=64, n_repeat=3, lambdas=lambdas_to_test)