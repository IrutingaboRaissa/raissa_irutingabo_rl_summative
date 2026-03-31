"""
NutriVision Decision Agent Environment
Reinforcement Learning environment for food recommendation and weight management.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import torch
    from environment.meal_cnn import MealEncoderCNN
except ImportError: # optional CNN tail (e.g. minimal installs)
    torch = None # type: ignore
    MealEncoderCNN = None # type: ignore


class NutriVisionEnv(gym.Env):
    """
    Custom Gymnasium environment for NutriVision RL agent.
    
    The agent learns to recommend foods that align with user's weight management goals.
    
    State Space (15-dim, Box 0..5000): 11 hand-crafted features (each normalized
    to [0,1] then scaled by 5000) plus 4 meal-CNN macro probabilities × 5000.
    
    Action Space: Discrete(4)
        0: Accept recommendation (log meal)
        1: Request lower-calorie alternative
        2: Request higher-protein alternative
        3: Skip meal (don't log)
    
    Reward: Positive when recommendation aligns with goal, negative otherwise.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = 20 # ~5 meals per day with 4 interactions each
        
        # African foods database with nutritional info (per standard portion)
        self.food_database = {
            0: {"name": "Jollof Rice", "cal": 280, "protein": 6, "carbs": 48, "fat": 5},
            1: {"name": "Ndole", "cal": 320, "protein": 18, "carbs": 25, "fat": 12},
            2: {"name": "Eru", "cal": 150, "protein": 8, "carbs": 12, "fat": 7},
            3: {"name": "Waakye", "cal": 220, "protein": 10, "carbs": 38, "fat": 3},
            4: {"name": "Ekwang", "cal": 350, "protein": 12, "carbs": 35, "fat": 15},
            5: {"name": "Palm-nut Soup", "cal": 280, "protein": 14, "carbs": 15, "fat": 18},
            6: {"name": "Suya", "cal": 320, "protein": 35, "carbs": 5, "fat": 18},
            7: {"name": "Injera with Doro", "cal": 310, "protein": 20, "carbs": 40, "fat": 8},
            8: {"name": "Ugali with Sukuma", "cal": 240, "protein": 8, "carbs": 42, "fat": 4},
            9: {"name": "Chapati", "cal": 280, "protein": 7, "carbs": 38, "fat": 12},
            10: {"name": "Fufu", "cal": 200, "protein": 4, "carbs": 44, "fat": 1},
            11: {"name": "Egusi Soup", "cal": 290, "protein": 16, "carbs": 18, "fat": 16},
            12: {"name": "Cassava Leaves", "cal": 120, "protein": 6, "carbs": 18, "fat": 3},
            13: {"name": "Pap with Beans", "cal": 260, "protein": 12, "carbs": 42, "fat": 5},
            14: {"name": "Yam Porridge", "cal": 270, "protein": 10, "carbs": 45, "fat": 6},
        }
        
        # Observation space: 15 continuous features
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(15,), dtype=np.float32
        )
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        self._meal_encoder = None
        if MealEncoderCNN is not None:
            self._meal_encoder = MealEncoderCNN()
            self._meal_encoder.eval()
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random goal: 0 = weight loss, 1 = weight gain, 2 = maintenance
        self.goal_type = self.np_random.integers(0, 3)
        
        # Daily targets based on goal
        if self.goal_type == 0: # Weight loss
            self.calorie_target = 2000
            self.protein_target = 150
            self.carbs_target = 150
            self.fats_target = 65
        elif self.goal_type == 1: # Weight gain
            self.calorie_target = 3000
            self.protein_target = 180
            self.carbs_target = 300
            self.fats_target = 100
        else: # Maintenance
            self.calorie_target = 2500
            self.protein_target = 160
            self.carbs_target = 220
            self.fats_target = 80
        
        # Current day intake
        self.daily_calories = 0
        self.daily_protein = 0
        self.daily_carbs = 0
        self.daily_fats = 0
        self.num_meals_logged = 0
        
        # Current meal being considered
        self.current_food_id = self.np_random.integers(0, len(self.food_database))
        self._sample_food()
        
        self.step_count = 0
        self.episode_reward = 0
        
        return self._get_observation(), {}
    
    def _sample_food(self):
        """Sample a random food from database."""
        self.current_food_id = self.np_random.integers(0, len(self.food_database))
        self.current_food = self.food_database[self.current_food_id].copy()
    
    def _norm_scale(self, value: float, hi: float) -> np.float32:
        """Map value to [0, 5000] via clip(value/hi, 0, 1) * 5000."""
        return np.float32(np.clip(float(value) / hi, 0.0, 1.0) * 5000.0)

    def _get_observation(self):
        """Return current observation: 11 scaled handcrafted + 4 CNN macro probs × 5000."""
        day_progress = self.step_count / max(self.max_steps, 1)
        handcrafted = np.array(
            [
                self._norm_scale(self.daily_calories, 8000.0),
                self._norm_scale(self.daily_protein, 250.0),
                self._norm_scale(self.daily_carbs, 500.0),
                self._norm_scale(self.daily_fats, 200.0),
                self._norm_scale(self.calorie_target, 3500.0),
                self._norm_scale(self.protein_target, 200.0),
                self._norm_scale(self.carbs_target, 350.0),
                self._norm_scale(self.fats_target, 120.0),
                self._norm_scale(float(self.goal_type), 2.0),
                self._norm_scale(day_progress, 1.0),
                self._norm_scale(float(self.num_meals_logged), float(self.max_steps)),
            ],
            dtype=np.float32,
        )

        if self._meal_encoder is not None and torch is not None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            macro = self._meal_encoder.encode_food_id(
                int(self.current_food_id), device=dev
            )
            tail = (macro.astype(np.float32) * 5000.0).astype(np.float32)
        else:
            tail = np.zeros(4, dtype=np.float32)

        obs = np.concatenate([handcrafted, tail]).astype(np.float32)
        return obs
    
    def _compute_reward(self, action):
        """
        Compute reward based on action alignment with goal.
        
        Reward structure:
        - Accept (0): +reward if meal aligns with goal, -reward otherwise
        - Request alternative (1,2): Small penalty but allows flexibility
        - Skip meal (3): Neutral
        """
        reward = 0.0
        
        if action == 0: # Accept recommendation
            # Calculate alignment metrics
            cal_ratio = (self.daily_calories + self.current_food["cal"]) / self.calorie_target
            protein_ratio = (self.daily_protein + self.current_food["protein"]) / self.protein_target
            carbs_ratio = (self.daily_carbs + self.current_food["carbs"]) / self.carbs_target
            fats_ratio = (self.daily_fats + self.current_food["fat"]) / self.fats_target
            
            # Ideal ratio is close to 1.0
            cal_diff = abs(cal_ratio - 1.0)
            protein_diff = abs(protein_ratio - 1.0)
            carbs_diff = abs(carbs_ratio - 1.0)
            fats_diff = abs(fats_ratio - 1.0)
            
            # Average deviation
            avg_deviation = (cal_diff + protein_diff + carbs_diff + fats_diff) / 4
            
            # Reward inversely proportional to deviation
            reward = 5.0 * (1.0 - min(avg_deviation, 1.0))
            
            # Goal-specific bonuses
            if self.goal_type == 0: # Weight loss: reward staying under targets
                if cal_ratio < 1.0 and protein_ratio > 0.8:
                    reward += 2.0
            elif self.goal_type == 1: # Weight gain: reward exceeding targets
                if cal_ratio > 0.8 and protein_ratio > 0.8:
                    reward += 2.0
            
        elif action == 1: # Request lower-calorie alternative
            reward = -0.5 # Small penalty for not accepting
            
        elif action == 2: # Request higher-protein alternative
            reward = -0.5 # Small penalty for not accepting
            
        elif action == 3: # Skip meal
            reward = 0.0 # Neutral
        
        return reward
    
    def step(self, action):
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = self._compute_reward(action)
        self.episode_reward += reward
        
        # Update state based on action
        if action == 0: # Accept recommendation
            self.daily_calories += self.current_food["cal"]
            self.daily_protein += self.current_food["protein"]
            self.daily_carbs += self.current_food["carbs"]
            self.daily_fats += self.current_food["fat"]
            self.num_meals_logged += 1
        
        # Sample next food
        self._sample_food()
        self.step_count += 1
        
        # Check terminal condition
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Additional termination: if daily targets exceeded by 50%
        if (self.daily_calories > self.calorie_target * 1.5 and 
            self.goal_type in [0, 2]):
            terminated = True
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {
            "food_name": self.current_food["name"],
            "episode_reward": self.episode_reward,
            "num_meals": self.num_meals_logged,
            "daily_calories": self.daily_calories,
            "daily_protein": self.daily_protein,
        }
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"\n--- Day Progress: {self.step_count}/{self.max_steps} ---")
            print(f"Goal: {['Weight Loss', 'Weight Gain', 'Maintenance'][self.goal_type]}")
            print(f"Daily Intake: {self.daily_calories:.0f} cal | Protein: {self.daily_protein:.1f}g")
            print(f"Targets: {self.calorie_target} cal | Protein: {self.protein_target:.1f}g")
            print(f"Current Food: {self.current_food['name']}")
            print(f"Meals Logged: {self.num_meals_logged}")
    
    def close(self):
        """Close environment."""
        pass
