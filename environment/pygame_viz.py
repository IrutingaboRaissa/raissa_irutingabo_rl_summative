"""
Pygame-based GUI Visualization for NutriVision Environment
Advanced 2D visualization with real-time agent state display
"""

import pygame
import numpy as np
from typing import Tuple, Dict
from environment.custom_env import NutriVisionEnv


class NutriVisionVisualizer:
    """Real-time 2D visualization of NutriVision environment using Pygame."""
    
    def __init__(self, width: int = 1400, height: int = 900):
        """Initialize Pygame visualization."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NutriVision RL - Agent Visualization")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors
        self.colors = {
            "bg": (245, 245, 245),
            "border": (50, 50, 50),
            "positive": (76, 175, 80),
            "negative": (244, 67, 54),
            "neutral": (33, 150, 243),
            "warning": (255, 152, 0),
            "text": (51, 51, 51),
            "light_text": (120, 120, 120),
        }
        
        self.running = True
    
    def draw_header(self, goal_text: str, step: int, max_steps: int):
        """Draw header section with goal and progress."""
        # Header background
        pygame.draw.rect(self.screen, (70, 130, 180), (0, 0, self.width, 80))
        
        # Title
        title = self.font_large.render("NutriVision RL Agent - Real-time Visualization", True, (255, 255, 255))
        self.screen.blit(title, (20, 15))
        
        # Goal and progress
        goal_label = self.font_medium.render(f"Goal: {goal_text}", True, (255, 255, 255))
        self.screen.blit(goal_label, (20, 50))
        
        progress_label = self.font_medium.render(f"Step: {step}/{max_steps}", True, (255, 255, 255))
        self.screen.blit(progress_label, (self.width - 250, 50))
    
    def draw_macros_section(self, daily: Dict, targets: Dict, x: int, y: int):
        """Draw macronutrient intake vs targets."""
        section_width = 400
        section_height = 280
        
        # Section background
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, section_width, section_height))
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, section_width, section_height), 2)
        
        # Title
        title = self.font_medium.render("Daily Macronutrient Intake", True, self.colors["text"])
        self.screen.blit(title, (x + 15, y + 10))
        
        # Macros to display
        macros = [
            ("Calories", daily["calories"], targets["calorie_target"], "kcal"),
            ("Protein", daily["protein"], targets["protein_target"], "g"),
            ("Carbs", daily["carbs"], targets["carbs_target"], "g"),
            ("Fats", daily["fats"], targets["fats_target"], "g"),
        ]
        
        bar_y = y + 50
        for name, actual, target, unit in macros:
            # Label
            label = self.font_small.render(f"{name}:", True, self.colors["text"])
            self.screen.blit(label, (x + 15, bar_y))
            
            # Progress bar background
            pygame.draw.rect(self.screen, (220, 220, 220), (x + 120, bar_y, 250, 20))
            
            # Progress bar fill
            ratio = min(actual / max(target, 1), 1.5) # Cap at 150%
            fill_width = int(250 * ratio)
            color = self.colors["positive"] if ratio < 1.0 else self.colors["warning"]
            pygame.draw.rect(self.screen, color, (x + 120, bar_y, fill_width, 20))
            
            # Text
            text = f"{actual:.0f} / {target:.0f} {unit}"
            text_surf = self.font_small.render(text, True, self.colors["light_text"])
            self.screen.blit(text_surf, (x + 125, bar_y + 2))
            
            bar_y += 50
    
    def draw_food_section(self, food_name: str, food_info: Dict, action: str, reward: float, 
                         x: int, y: int):
        """Draw current food and recommended action."""
        section_width = 400
        section_height = 280
        
        # Section background
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, section_width, section_height))
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, section_width, section_height), 2)
        
        # Title
        title = self.font_medium.render("Current Food Recommendation", True, self.colors["text"])
        self.screen.blit(title, (x + 15, y + 10))
        
        # Food name (highlight)
        food_label = self.font_large.render(food_name, True, (25, 118, 210))
        self.screen.blit(food_label, (x + 15, y + 50))
        
        # Nutritional info
        nutrition_y = y + 100
        nutrition = [
            f"Calories: {food_info['cal']} kcal",
            f"Protein: {food_info['protein']}g",
            f"Carbs: {food_info['carbs']}g",
            f"Fats: {food_info['fat']}g",
        ]
        
        for info in nutrition:
            text_surf = self.font_small.render(info, True, self.colors["text"])
            self.screen.blit(text_surf, (x + 15, nutrition_y))
            nutrition_y += 30
        
        # Recommended action
        action_color = self.colors["positive"] if action == "Accept" else self.colors["warning"]
        action_label = self.font_medium.render(f"Recommended: {action}", True, action_color)
        self.screen.blit(action_label, (x + 15, y + 230))
    
    def draw_reward_section(self, episode_reward: float, step_reward: float, 
                           num_meals: int, x: int, y: int):
        """Draw reward and performance metrics."""
        section_width = 350
        section_height = 280
        
        # Section background
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, section_width, section_height))
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, section_width, section_height), 2)
        
        # Title
        title = self.font_medium.render("Performance Metrics", True, self.colors["text"])
        self.screen.blit(title, (x + 15, y + 10))
        
        # Metrics
        metrics_y = y + 60
        metrics = [
            ("Episode Reward", episode_reward, self.colors["neutral"]),
            ("Step Reward", step_reward, self.colors["positive"] if step_reward > 0 else self.colors["negative"]),
            ("Meals Logged", num_meals, self.colors["text"]),
        ]
        
        for label, value, color in metrics:
            label_text = self.font_small.render(f"{label}:", True, self.colors["text"])
            self.screen.blit(label_text, (x + 15, metrics_y))
            
            if isinstance(value, float):
                value_text = f"{value:+.2f}"
            else:
                value_text = str(value)
            
            value_surf = self.font_large.render(value_text, True, color)
            self.screen.blit(value_surf, (x + 15, metrics_y + 25))
            
            metrics_y += 80
    
    def draw_action_buttons(self, y: int):
        """Draw action legend."""
        label = self.font_medium.render("Actions:", True, self.colors["text"])
        self.screen.blit(label, (20, y))
        
        actions = [
            ("0: Accept", self.colors["positive"]),
            ("1: Lower Cal", self.colors["warning"]),
            ("2: Higher Protein", self.colors["neutral"]),
            ("3: Skip", self.colors["negative"]),
        ]
        
        x = 150
        for action, color in actions:
            text = self.font_small.render(action, True, color)
            self.screen.blit(text, (x, y))
            x += 160
    
    def render_episode(self, env: NutriVisionEnv, action: int = None, 
                      reward: float = 0.0, show_fps: bool = True):
        """Render a single frame of the environment."""
        # Clear screen
        self.screen.fill(self.colors["bg"])
        
        # Extract environment state
        goal_types = ["Weight Loss", "Weight Gain", "Maintenance"]
        goal_text = goal_types[env.goal_type]
        
        daily = {
            "calories": env.daily_calories,
            "protein": env.daily_protein,
            "carbs": env.daily_carbs,
            "fats": env.daily_fats,
        }
        
        targets = {
            "calorie_target": env.calorie_target,
            "protein_target": env.protein_target,
            "carbs_target": env.carbs_target,
            "fats_target": env.fats_target,
        }
        
        action_names = ["Accept", "Lower Cal", "Higher Protein", "Skip"]
        action_text = action_names[action] if action is not None else "None"
        
        # Draw sections
        self.draw_header(goal_text, env.step_count, env.max_steps)
        self.draw_macros_section(daily, targets, 20, 110)
        self.draw_food_section(
            env.current_food["name"],
            env.current_food,
            action_text,
            reward,
            450, 110
        )
        self.draw_reward_section(
            env.episode_reward,
            reward,
            env.num_meals_logged,
            900, 110
        )
        
        # Draw action legend
        self.draw_action_buttons(450)
        
        # Draw FPS
        if show_fps:
            fps_text = self.font_small.render(f"FPS: {int(self.clock.get_fps())}", True, self.colors["light_text"])
            self.screen.blit(fps_text, (self.width - 150, self.height - 30))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30) # 30 FPS
    
    def play_episode_interactive(self, env: NutriVisionEnv):
        """Run an interactive episode with human control."""
        obs, _ = env.reset()
        action_map = {
            pygame.K_0: 0, # Accept
            pygame.K_1: 1, # Lower calorie
            pygame.K_2: 2, # Higher protein
            pygame.K_3: 3, # Skip
        }
        
        print("\n" + "="*80)
        print("INTERACTIVE EPISODE - Control Agent with Keys 0-3")
        print("="*80)
        print("0: Accept recommendation")
        print("1: Request lower-calorie alternative")
        print("2: Request higher-protein alternative")
        print("3: Skip meal")
        print("Close window or press ESC to stop")
        print("="*80 + "\n")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in action_map:
                        action = action_map[event.key]
                        obs, reward, terminated, truncated, info = env.step(action)
                        self.render_episode(env, action, reward)
                        
                        print(f"Action: {['Accept', 'Lower Cal', 'Higher Protein', 'Skip'][action]} | "
                              f"Reward: {reward:+.2f} | Food: {info['food_name']}")
                        
                        if terminated or truncated:
                            print(f"\nEpisode ended! Total reward: {info['episode_reward']:+.2f}")
                            return
            
            self.render_episode(env)
        
        self.close()
    
    def record_episode_video(
        self,
        env: NutriVisionEnv,
        model,
        algorithm: str,
        output_path: str = "visualizations/agent_demo.avi",
        *,
        show_fps: bool = False,
        verbose: bool = True,
        max_frames: int | None = None,
    ):
        """Record a single episode (using the trained model) into a video file.

        Uses MJPG in an AVI container by default — plays reliably on Windows;
        short MP4 files from OpenCV often fail in some players.
        """
        import cv2
        
        obs, _ = env.reset()
        
        # MJPG + AVI is the most portable combo with OpenCV on Windows.
        path_lower = output_path.lower()
        if path_lower.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.width, self.height))
        if not out.isOpened():
            raise RuntimeError(
                f"Could not open VideoWriter for {output_path!r}. "
                "Try .avi with MJPG or install a full OpenCV build."
            )
        
        frame_count = 0
        
        while True:
            # Get action from model
            if algorithm == "reinforce":
                import torch
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = model(state_tensor)
                action = torch.argmax(probs, dim=1).item()
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame
            self.render_episode(env, action, reward, show_fps=show_fps)

            if verbose:
                action_names = ["Accept", "Lower Cal", "Higher Protein", "Skip"]
                food_name = info.get("food_name", "?")
                print(
                    f"Step {env.step_count:02d} | "
                    f"Action: {action_names[action]} | "
                    f"Reward: {reward:+.2f} | "
                    f"Food: {food_name}"
                )
            
            # Capture frame
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
            frame_count += 1

            if max_frames is not None and frame_count >= max_frames:
                break
            
            if terminated or truncated:
                break
        
        out.release()
        print(f"[OK] Video saved to {output_path} ({frame_count} frames)")
    
    def close(self):
        """Close Pygame."""
        pygame.quit()


def demo_visualization():
    """Demo function to show interactive visualization."""
    env = NutriVisionEnv()
    visualizer = NutriVisionVisualizer()
    
    try:
        visualizer.play_episode_interactive(env)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        visualizer.close()


if __name__ == "__main__":
    demo_visualization()
