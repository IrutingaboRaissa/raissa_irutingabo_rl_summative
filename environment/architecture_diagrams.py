"""
Architecture Diagram Generator for NutriVision
Creates visual diagrams of the system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_environment_architecture():
    """Draw the NutriVision environment architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, "NutriVision RL - System Architecture", 
            fontsize=20, fontweight='bold', ha='center')
    
    # Color scheme
    color_agent = "#FFB6C1"
    color_env = "#87CEEB"
    color_reward = "#90EE90"
    color_state = "#FFD700"
    
    # --- AGENT BOX ---
    agent_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_agent, linewidth=2)
    ax.add_patch(agent_box)
    ax.text(1.5, 7.5, "RL Agent", fontsize=13, fontweight='bold', ha='center')
    ax.text(1.5, 7.0, "Policy/Value Fn", fontsize=10, ha='center')
    
    # --- ENVIRONMENT BOX ---
    env_box = FancyBboxPatch((4.5, 6.5), 2, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color_env, linewidth=2)
    ax.add_patch(env_box)
    ax.text(5.5, 7.5, "NutriVision Env", fontsize=13, fontweight='bold', ha='center')
    ax.text(5.5, 7.0, "Gymnasium", fontsize=10, ha='center')
    
    # --- STATE BOX ---
    state_box = FancyBboxPatch((8, 6.5), 1.8, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=color_state, linewidth=2)
    ax.add_patch(state_box)
    ax.text(8.9, 7.3, "Observation", fontsize=11, fontweight='bold', ha='center')
    ax.text(8.9, 6.9, "(15-dim)", fontsize=9, ha='center')
    
    # --- ACTION FLOW ---
    arrow1 = FancyArrowPatch((2.5, 7.2), (4.5, 7.2),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='red')
    ax.add_patch(arrow1)
    ax.text(3.5, 7.5, "Action", fontsize=10, fontweight='bold', color='red')
    
    # --- STATE FLOW ---
    arrow2 = FancyArrowPatch((6.5, 7.2), (8, 7.2),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='blue')
    ax.add_patch(arrow2)
    ax.text(7.2, 7.5, "State", fontsize=10, fontweight='bold', color='blue')
    
    # --- REWARD BOX ---
    reward_box = FancyBboxPatch((4.5, 4.5), 2, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_reward, linewidth=2)
    ax.add_patch(reward_box)
    ax.text(5.5, 5.5, "Reward", fontsize=13, fontweight='bold', ha='center')
    ax.text(5.5, 5.0, "Alignment Score", fontsize=10, ha='center')
    
    # Reward arrow
    arrow3 = FancyArrowPatch((5.5, 6.5), (5.5, 6.0),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='green')
    ax.add_patch(arrow3)
    ax.text(5.9, 6.2, "Reward", fontsize=10, fontweight='bold', color='green')
    
    # Feedback arrow
    arrow4 = FancyArrowPatch((5.5, 4.5), (1.5, 6.5),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='purple', linestyle='dashed')
    ax.add_patch(arrow4)
    ax.text(3, 5.2, "Learn", fontsize=10, fontweight='bold', color='purple')
    
    # --- STATE SPACE DETAILS ---
    state_details = [
        "Observation Space (15-dim):",
        "• Daily macros: cal, protein, carbs, fats",
        "• Daily targets: cal, protein, carbs, fats",
        "• Goal type: loss, gain, maintenance",
        "• Current food: cal, protein, carbs, fats",
        "• Day progress & meals logged"
    ]
    
    y_pos = 3.8
    ax.text(0.3, 3.8, "\n".join(state_details), fontsize=9, 
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- ACTION SPACE DETAILS ---
    action_details = [
        "Action Space (Discrete-4):",
        "0: Accept recommendation",
        "1: Request lower-calorie",
        "2: Request higher-protein",
        "3: Skip meal"
    ]
    
    ax.text(5.2, 3.8, "\n".join(action_details), fontsize=9,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # --- REWARD STRUCTURE ---
    reward_details = [
        "Reward Structure:",
        "Accept: +5.0 if aligned,",
        " scaled by deviation",
        "Request: -0.5 penalty",
        "Skip: 0.0 (neutral)"
    ]
    
    ax.text(8.0, 3.8, "\n".join(reward_details), fontsize=9,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # --- ENVIRONMENT COMPONENTS ---
    components = [
        "Environment Components:",
        "• 15 African foods database",
        "• Weight management goals (3 types)",
        "• Nutritional profiles per food",
        "• Daily macro tracking",
        "• Termination: 20 steps or overage"
    ]
    
    ax.text(0.3, 0.8, "\n".join(components), fontsize=9,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # --- ALGORITHMS ---
    algorithms = [
        "Algorithms Trained:",
        "• DQN (Value-Based)",
        "• REINFORCE (Policy Gradient)",
        "• PPO (Proximal Policy Opt.)",
        "• A2C (Actor-Critic)"
    ]
    
    ax.text(3.5, 0.8, "\n".join(algorithms), fontsize=9,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("visualizations/system_architecture.png", dpi=300, bbox_inches='tight')
    print("[OK] System architecture diagram saved to visualizations/system_architecture.png")
    plt.close()


def draw_training_pipeline():
    """Draw the training pipeline architecture."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, "NutriVision RL - Training Pipeline", 
            fontsize=20, fontweight='bold', ha='center')
    
    # Stage 1: Environment
    stage1 = FancyBboxPatch((0.5, 7.5), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(stage1)
    ax.text(1.75, 8.3, "1. Environment", fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 7.9, "NutriVisionEnv", fontsize=9, ha='center')
    
    # Arrow
    arrow = FancyArrowPatch((3.0, 8.1), (4.0, 8.1),
                           arrowstyle='->', mutation_scale=30, linewidth=2)
    ax.add_patch(arrow)
    
    # Stage 2: Data Collection
    stage2 = FancyBboxPatch((4.0, 7.5), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(stage2)
    ax.text(5.25, 8.3, "2. Data Collection", fontsize=12, fontweight='bold', ha='center')
    ax.text(5.25, 7.9, "Episodes & Trajectories", fontsize=9, ha='center')
    
    # Arrow
    arrow = FancyArrowPatch((6.5, 8.1), (7.5, 8.1),
                           arrowstyle='->', mutation_scale=30, linewidth=2)
    ax.add_patch(arrow)
    
    # Stage 3: Model Training
    stage3 = FancyBboxPatch((7.5, 7.5), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(stage3)
    ax.text(8.75, 8.3, "3. Model Training", fontsize=12, fontweight='bold', ha='center')
    ax.text(8.75, 7.9, "Policy/Value Update", fontsize=9, ha='center')
    
    # Arrow
    arrow = FancyArrowPatch((10.0, 8.1), (11.0, 8.1),
                           arrowstyle='->', mutation_scale=30, linewidth=2)
    ax.add_patch(arrow)
    
    # Stage 4: Evaluation
    stage4 = FancyBboxPatch((11.0, 7.5), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#FFF3E0', linewidth=2)
    ax.add_patch(stage4)
    ax.text(12.25, 8.3, "4. Evaluation", fontsize=12, fontweight='bold', ha='center')
    ax.text(12.25, 7.9, "Test Reward", fontsize=9, ha='center')
    
    # Arrow to loop
    arrow = FancyArrowPatch((12.5, 7.5), (12.5, 6.5),
                           arrowstyle='->', mutation_scale=30, linewidth=2)
    ax.add_patch(arrow)
    
    # Feedback loop
    arrow = FancyArrowPatch((12.5, 6.5), (1.75, 6.5),
                           arrowstyle='->', mutation_scale=30, linewidth=2,
                           connectionstyle="arc3,rad=.5", linestyle='dashed')
    ax.add_patch(arrow)
    ax.text(7, 5.9, "Repeat for next episode", fontsize=10, ha='center', style='italic')
    
    # --- ALGORITHM BOXES ---
    algorithms_y = 4.5
    algorithms = [
        ("DQN\n(Value-Based)", "#FFEBEE"),
        ("REINFORCE\n(Policy Gradient)", "#FCE4EC"),
        ("PPO\n(Policy Gradient)", "#F3E5F5"),
        ("A2C\n(Actor-Critic)", "#E8F5E9")
    ]
    
    x_positions = [1.5, 4.5, 7.5, 10.5]
    
    for (algo_name, color), x_pos in zip(algorithms, x_positions):
        box = FancyBboxPatch((x_pos - 1, algorithms_y - 0.8), 2, 1.6,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos, algorithms_y + 0.3, algo_name, fontsize=11, fontweight='bold', 
               ha='center', va='center')
    
    ax.text(8, algorithms_y + 1.5, "4 Algorithms Trained with 11 Hyperparameter Configurations Each",
           fontsize=12, fontweight='bold', ha='center')
    
    # --- HYPERPARAMETER TUNING ---
    hp_details = [
        "Hyperparameter Tuning:",
        "• Learning Rate: 1e-5 to 1e-2",
        "• Batch Size: 16 to 128",
        "• Gamma (discount): 0.95 to 0.999",
        "• Buffer Size: 50k to 100k",
        "• Entropy Coefficient: 0.0 to 0.05",
        "• Total: 44 configurations"
    ]
    
    ax.text(0.5, 2.0, "\n".join(hp_details), fontsize=10,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # --- OUTPUT ---
    output_details = [
        "Training Outputs:",
        "• Trained models (44 total)",
        "• Performance metrics",
        "• Learning curves",
        "• Hyperparameter rankings",
        "• Best model checkpoint"
    ]
    
    ax.text(5.0, 2.0, "\n".join(output_details), fontsize=10,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # --- EVALUATION ---
    eval_details = [
        "Evaluation Metrics:",
        "• Mean Episode Reward",
        "• Std Dev of Reward",
        "• Convergence Speed",
        "• Generalization",
        "• Agent Behavior"
    ]
    
    ax.text(10.0, 2.0, "\n".join(eval_details), fontsize=10,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("visualizations/training_pipeline.png", dpi=300, bbox_inches='tight')
    print("[OK] Training pipeline diagram saved to visualizations/training_pipeline.png")
    plt.close()


def draw_agent_decision_flow():
    """Draw the agent decision-making flow diagram."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, "NutriVision Agent - Decision Flow", 
            fontsize=20, fontweight='bold', ha='center')
    
    # Start
    start = patches.Circle((7, 8.5), 0.3, color='green', ec='black', linewidth=2)
    ax.add_patch(start)
    ax.text(7, 8.5, '', fontsize=12, ha='center', va='center', fontweight='bold')
    
    arrow = FancyArrowPatch((7, 8.2), (7, 7.5),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Observe state
    obs_box = FancyBboxPatch((5.5, 6.8), 3, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(obs_box)
    ax.text(7, 7.1, "Observe State (15-dim)", fontsize=11, fontweight='bold', ha='center')
    
    arrow = FancyArrowPatch((7, 6.8), (7, 6.2),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Process through network
    network_box = FancyBboxPatch((5, 5.5), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(network_box)
    ax.text(7, 5.8, "Feed through Neural Network", fontsize=11, fontweight='bold', ha='center')
    
    arrow = FancyArrowPatch((7, 5.5), (7, 4.9),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Get action distribution
    dist_box = FancyBboxPatch((4.5, 4.2), 5, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(dist_box)
    ax.text(7, 4.5, "Get Action Probabilities/Values", fontsize=11, fontweight='bold', ha='center')
    
    arrow = FancyArrowPatch((7, 4.2), (7, 3.6),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Select action
    action_box = FancyBboxPatch((4, 2.9), 6, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='#FFF3E0', linewidth=2)
    ax.add_patch(action_box)
    ax.text(7, 3.2, "Select Action (sample or greedy)", fontsize=11, fontweight='bold', ha='center')
    
    # Actions split
    actions = ["Accept", "Lower Cal", "Higher Protein", "Skip"]
    action_x = [2, 4.5, 7, 9.5]
    
    for action, x in zip(actions, action_x):
        arrow = FancyArrowPatch((7, 2.9), (x, 2.2),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5)
        ax.add_patch(arrow)
        
        action_box_small = FancyBboxPatch((x - 0.7, 1.5), 1.4, 0.6,
                                         boxstyle="round,pad=0.05",
                                         edgecolor='black', facecolor='#E0E0E0', linewidth=1)
        ax.add_patch(action_box_small)
        ax.text(x, 1.8, action, fontsize=9, fontweight='bold', ha='center')
    
    # All actions reconverge
    for x in action_x:
        arrow = FancyArrowPatch((x, 1.5), (7, 0.9),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5,
                               connectionstyle="arc3,rad=.2")
        ax.add_patch(arrow)
    
    # Execute and receive reward
    exec_box = FancyBboxPatch((4.5, 0.2), 5, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(exec_box)
    ax.text(7, 0.5, "Execute Action & Receive Reward", fontsize=11, fontweight='bold', ha='center')
    
    # Side information boxes
    info_boxes = [
        ("Exploration vs Exploitation:\nBalance between trying new\nactions and exploiting best", 0.5, 5.0, '#E1F5FE'),
        ("Reward Computation:\nBased on macro alignment\nwith daily targets", 10.5, 5.0, '#FFF9C4'),
        ("Learning Signal:\nReward used to update\npolicy/value function", 0.5, 0.5, '#F0F4C3'),
    ]
    
    for text, x, y, color in info_boxes:
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig("visualizations/decision_flow.png", dpi=300, bbox_inches='tight')
    print("[OK] Decision flow diagram saved to visualizations/decision_flow.png")
    plt.close()


def main():
    """Generate all architecture diagrams."""
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING ARCHITECTURE DIAGRAMS")
    print("="*80 + "\n")
    
    try:
        draw_environment_architecture()
    except Exception as e:
        print(f"[WARN] Error generating environment architecture: {e}")
    
    try:
        draw_training_pipeline()
    except Exception as e:
        print(f"[WARN] Error generating training pipeline: {e}")
    
    try:
        draw_agent_decision_flow()
    except Exception as e:
        print(f"[WARN] Error generating decision flow: {e}")
    
    print("\n" + "="*80)
    print("[OK] All diagrams generated successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
