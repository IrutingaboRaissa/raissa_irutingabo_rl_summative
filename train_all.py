"""
Master Training Script for NutriVision RL Agents
Trains all 4 algorithms with full hyperparameter tuning
"""

import sys
import time
import json
import os
from datetime import datetime

from training.dqn_training import DQNTrainer
from training.pg_training import PPOTrainer, A2CTrainer
from training.reinforce_training import REINFORCETrainer
from environment.rendering import EnvironmentVisualizer


def main():
    """Train all algorithms."""
    
    print("\n" + "="*100)
    print(" "*25 + "NUTRIVISION RL TRAINING - MASTER SCRIPT")
    print("="*100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # Runtime knobs (can be overridden via environment variables).
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "50000"))
    reinforce_episodes = int(os.getenv("REINFORCE_EPISODES", "500"))
    print(f"TOTAL_TIMESTEPS={total_timesteps} | REINFORCE_EPISODES={reinforce_episodes}")

    start_time = time.time()
    all_results = {}
    
    # ============================================================================
    # 1. TRAIN DQN
    # ============================================================================
    print("\n" + "="*100)
    print("PHASE 1: DQN TRAINING (Value-Based Method)")
    print("="*100)
    
    dqn_start = time.time()
    try:
        dqn_trainer = DQNTrainer()
        dqn_results = dqn_trainer.train_all_configs(total_timesteps=total_timesteps)
        all_results["dqn"] = {
            "status": "success",
            "results": {k: {"mean_reward": v["mean_reward"], "std_reward": v["std_reward"]} 
                       for k, v in dqn_results.items()},
            "training_time": time.time() - dqn_start
        }
        print(f"[OK] DQN training completed in {all_results['dqn']['training_time']:.1f}s")
    except Exception as e:
        print(f"[FAIL] DQN training failed: {e}")
        all_results["dqn"] = {"status": "failed", "error": str(e)}
    
    # ============================================================================
    # 2. TRAIN PPO
    # ============================================================================
    print("\n" + "="*100)
    print("PHASE 2: PPO TRAINING (Proximal Policy Optimization)")
    print("="*100)
    
    ppo_start = time.time()
    try:
        ppo_trainer = PPOTrainer()
        ppo_results = ppo_trainer.train_all_configs(total_timesteps=total_timesteps)
        all_results["ppo"] = {
            "status": "success",
            "results": {k: {"mean_reward": v["mean_reward"], "std_reward": v["std_reward"]} 
                       for k, v in ppo_results.items()},
            "training_time": time.time() - ppo_start
        }
        print(f"[OK] PPO training completed in {all_results['ppo']['training_time']:.1f}s")
    except Exception as e:
        print(f"[FAIL] PPO training failed: {e}")
        all_results["ppo"] = {"status": "failed", "error": str(e)}
    
    # ============================================================================
    # 3. TRAIN A2C
    # ============================================================================
    print("\n" + "="*100)
    print("PHASE 3: A2C TRAINING (Advantage Actor-Critic)")
    print("="*100)
    
    a2c_start = time.time()
    try:
        a2c_trainer = A2CTrainer()
        a2c_results = a2c_trainer.train_all_configs(total_timesteps=total_timesteps)
        all_results["a2c"] = {
            "status": "success",
            "results": {k: {"mean_reward": v["mean_reward"], "std_reward": v["std_reward"]} 
                       for k, v in a2c_results.items()},
            "training_time": time.time() - a2c_start
        }
        print(f"[OK] A2C training completed in {all_results['a2c']['training_time']:.1f}s")
    except Exception as e:
        print(f"[FAIL] A2C training failed: {e}")
        all_results["a2c"] = {"status": "failed", "error": str(e)}
    
    # ============================================================================
    # 4. TRAIN REINFORCE
    # ============================================================================
    print("\n" + "="*100)
    print("PHASE 4: REINFORCE TRAINING (Vanilla Policy Gradient)")
    print("="*100)
    
    reinforce_start = time.time()
    try:
        reinforce_trainer = REINFORCETrainer(log_dir="models/pg")
        reinforce_results = reinforce_trainer.train_all_configs(num_episodes=reinforce_episodes)
        all_results["reinforce"] = {
            "status": "success",
            "results": {k: {"mean_reward": v["mean_reward"], "std_reward": v["std_reward"]} 
                       for k, v in reinforce_results.items()},
            "training_time": time.time() - reinforce_start
        }
        print(f"[OK] REINFORCE training completed in {all_results['reinforce']['training_time']:.1f}s")
    except Exception as e:
        print(f"[FAIL] REINFORCE training failed: {e}")
        all_results["reinforce"] = {"status": "failed", "error": str(e)}
    
    # ============================================================================
    # SUMMARY AND ANALYSIS
    # ============================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE - SUMMARY REPORT")
    print("="*100)
    print(f"Total Training Time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find best overall configuration
    best_overall = None
    best_reward = -float('inf')
    
    print("\n" + "="*100)
    print("ALGORITHM PERFORMANCE RANKINGS")
    print("="*100)
    
    algorithm_stats = {}
    
    for algo, data in all_results.items():
        if data["status"] == "success":
            rewards = [v["mean_reward"] for v in data["results"].values()]
            algorithm_stats[algo] = {
                "mean": sum(rewards) / len(rewards),
                "max": max(rewards),
                "min": min(rewards),
                "count": len(rewards),
            }
            
            print(f"\n{algo.upper()}:")
            print(f" Configurations Trained: {algorithm_stats[algo]['count']}")
            print(f" Average Reward: {algorithm_stats[algo]['mean']:.3f}")
            print(f" Best Config Reward: {algorithm_stats[algo]['max']:.3f}")
            print(f" Worst Config Reward: {algorithm_stats[algo]['min']:.3f}")
            print(f" Training Time: {data['training_time']/60:.1f} minutes")
            
            # Track best overall
            if algorithm_stats[algo]['max'] > best_reward:
                best_reward = algorithm_stats[algo]['max']
                best_overall = algo
    
    print("\n" + "="*100)
    if best_overall is not None:
        print(f"BEST OVERALL ALGORITHM: {best_overall.upper()} (Reward: {best_reward:.3f})")
    else:
        print("BEST OVERALL ALGORITHM: N/A (no successful runs)")
    print("="*100)
    
    # Save master results
    master_results = {
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": total_time,
        "total_training_time_hours": total_time / 3600,
        "algorithms": all_results,
        "algorithm_stats": algorithm_stats,
        "best_overall_algorithm": best_overall,
        "best_overall_reward": best_reward,
    }
    
    with open("models/master_training_results.json", "w") as f:
        json.dump(master_results, f, indent=2)
    
    print(f"\n[OK] Master results saved to models/master_training_results.json")
    
    # Print detailed rankings
    print("\n" + "="*100)
    print("DETAILED CONFIGURATION RANKINGS BY ALGORITHM")
    print("="*100)
    
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        if algo in all_results and all_results[algo]["status"] == "success":
            print(f"\n{algo.upper()} - Top 5 Configurations:")
            configs = sorted(
                all_results[algo]["results"].items(),
                key=lambda x: x[1]["mean_reward"],
                reverse=True
            )[:5]
            
            for rank, (config, stats) in enumerate(configs, 1):
                print(f" {rank}. {config:30s} | Reward: {stats['mean_reward']:7.3f} ± {stats['std_reward']:6.3f}")
    
    print("\n" + "="*100)
    print("NEXT STEPS:")
    print("="*100)
    print("1. Review detailed results in models/[algorithm]/[algorithm]_summary.json")
    print("2. Run playback: python main.py")
    print("3. Generate visualizations: python -c \"from environment.rendering import *; ..\"")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
