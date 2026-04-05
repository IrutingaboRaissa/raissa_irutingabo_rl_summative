"""
JSON API Module for NutriVision RL Agent
Enables integration with web/mobile frontends
"""

import json
import os
from typing import Dict, Any, List
import numpy as np
from datetime import datetime


class NutriVisionAPI:
    """API wrapper for NutriVision RL agent with JSON serialization."""
    
    def __init__(self, model=None, algorithm: str = "ppo"):
        """Initialize API with loaded model."""
        self.model = model
        self.algorithm = algorithm
        self.session_history = []
        self.current_state = None
    
    def get_recommendation(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get food recommendation from agent.
        
        Args:
            user_state: {
                "daily_calories": float,
                "daily_protein": float,
                "daily_carbs": float,
                "daily_fats": float,
                "calorie_target": float,
                "protein_target": float,
                "carbs_target": float,
                "fats_target": float,
                "goal_type": int, # 0=loss, 1=gain, 2=maintenance
                "current_food": {
                    "cal": float,
                    "protein": float,
                    "carbs": float,
                    "fat": float
                },
                "day_progress": float,
                "num_meals_logged": int
            }
        
        Returns:
            {
                "status": "success",
                "action": int,
                "action_name": str,
                "food_recommendation": {
                    "action": str,
                    "explanation": str,
                    "confidence": float
                },
                "timestamp": str,
                "session_id": str
            }
        """
        try:
            # Prepare observation
            obs = np.array([
                user_state["daily_calories"],
                user_state["daily_protein"],
                user_state["daily_carbs"],
                user_state["daily_fats"],
                user_state["calorie_target"],
                user_state["protein_target"],
                user_state["carbs_target"],
                user_state["fats_target"],
                float(user_state["goal_type"]),
                user_state["current_food"]["cal"],
                user_state["current_food"]["protein"],
                user_state["current_food"]["carbs"],
                user_state["current_food"]["fat"],
                user_state["day_progress"],
                float(user_state["num_meals_logged"]),
            ], dtype=np.float32)
            
            self.current_state = obs
            
            # Get action from model
            if self.algorithm == "reinforce":
                import torch
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = self.model(state_tensor)
                action = torch.argmax(probs, dim=1).item()
                confidence = float(torch.max(probs).item())
            else:
                action, _ = self.model.predict(obs, deterministic=True)
                confidence = 0.85 # Default confidence for SB3 models
            
            # Generate explanation
            action_names = ["Accept", "Request Lower Calorie", "Request Higher Protein", "Skip"]
            explanation = self._generate_explanation(action, user_state)
            
            response = {
                "status": "success",
                "action": int(action),
                "action_name": action_names[action],
                "food_recommendation": {
                    "action": action_names[action],
                    "explanation": explanation,
                    "confidence": float(confidence)
                },
                "timestamp": datetime.now().isoformat(),
                "session_id": self._get_session_id()
            }
            
            # Log to history
            self.session_history.append(response)
            
            return response
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_explanation(self, action: int, user_state: Dict) -> str:
        """Generate human-readable explanation for action."""
        goal_names = ["weight loss", "weight gain", "maintenance"]
        goal = goal_names[user_state["goal_type"]]
        
        cal_ratio = (user_state["daily_calories"] + user_state["current_food"]["cal"]) / user_state["calorie_target"]
        protein_ratio = (user_state["daily_protein"] + user_state["current_food"]["protein"]) / user_state["protein_target"]
        
        if action == 0: # Accept
            if cal_ratio < 1.0 and protein_ratio > 0.8:
                return f"Great choice for {goal}! This meal aligns well with your daily targets."
            elif cal_ratio > 1.2:
                return f"This is higher in calories, but could work if you need more energy."
            else:
                return f"This meal fits reasonably well with your {goal} goals."
        
        elif action == 1: # Lower calorie
            return f"This meal might be too high in calories for {goal}. Consider a lower-calorie alternative."
        
        elif action == 2: # Higher protein
            return f"Good protein opportunity! Request higher-protein version to maximize nutritional alignment."
        
        elif action == 3: # Skip
            return f"Skip this meal - it doesn't align well with your {goal} objectives today."
        
        return "Neutral recommendation"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session recommendations."""
        if not self.session_history:
            return {"status": "no_history", "recommendations_count": 0}
        
        actions = [h.get("action", -1) for h in self.session_history if h.get("status") == "success"]
        action_counts = {
            "accept": actions.count(0),
            "lower_cal": actions.count(1),
            "higher_protein": actions.count(2),
            "skip": actions.count(3),
        }
        
        avg_confidence = np.mean([
            h.get("food_recommendation", {}).get("confidence", 0)
            for h in self.session_history if h.get("status") == "success"
        ]) if self.session_history else 0
        
        return {
            "status": "success",
            "total_recommendations": len(self.session_history),
            "successful_recommendations": len([h for h in self.session_history if h.get("status") == "success"]),
            "action_breakdown": action_counts,
            "average_confidence": float(avg_confidence),
            "session_start": self.session_history[0].get("timestamp") if self.session_history else None,
            "session_end": self.session_history[-1].get("timestamp") if self.session_history else None,
        }
    
    def export_session_json(self, filepath: str = "session_log.json") -> str:
        """Export session history to JSON file."""
        export_data = {
            "session_info": self.get_session_summary(),
            "algorithm": self.algorithm,
            "recommendations": self.session_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about the loaded model."""
        return {
            "algorithm": self.algorithm,
            "model_type": self.model.__class__.__name__ if hasattr(self.model, '__class__') else "Unknown",
            "input_size": 15,
            "output_size": 4,
            "description": "NutriVision Food Recommendation Agent",
            "version": "1.0.0",
            "supported_goals": ["weight_loss", "weight_gain", "maintenance"],
            "num_foods": 15,
            "african_foods": [
                "Jollof Rice", "Ndole", "Eru", "Waakye", "Ekwang", "Palm-nut Soup",
                "Suya", "Injera with Doro", "Ugali with Sukuma", "Chapati",
                "Fufu", "Egusi Soup", "Cassava Leaves", "Pap with Beans", "Yam Porridge"
            ]
        }
    
    def _get_session_id(self) -> str:
        """Get or create session ID."""
        if not hasattr(self, '_session_id'):
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_id
    
    def to_json(self) -> str:
        """Serialize API state to JSON."""
        return json.dumps({
            "algorithm": self.algorithm,
            "session_summary": self.get_session_summary(),
            "model_info": self.get_model_info(),
        }, indent=2)

class NutriVisionEndpoints:
    """REST API endpoint handlers."""
    
    def __init__(self, api: NutriVisionAPI):
        self.api = api
    
    def handle_recommendation_request(self, request_json: str) -> str:
        """Handle incoming recommendation request."""
        try:
            data = json.loads(request_json)
            response = self.api.get_recommendation(data)
            return json.dumps(response)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    def handle_session_summary_request(self) -> str:
        """Handle session summary request."""
        return json.dumps(self.api.get_session_summary())
    
    def handle_model_info_request(self) -> str:
        """Handle model info request."""
        return json.dumps(self.api.get_model_info())
    
    def handle_export_request(self, filepath: str = "session_log.json") -> str:
        """Handle export request."""
        try:
            result = self.api.export_session_json(filepath)
            return json.dumps({"status": "success", "filepath": result})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})


# Example usage
example_usage = """
# Example: Using NutriVision API with Flask

from flask import Flask, request, jsonify
from nutrivision_api import NutriVisionAPI, NutriVisionEndpoints
from stable_baselines3 import PPO

app = Flask(__name__)

# Load model
model = PPO.load("models/ppo/ppo_default")
api = NutriVisionAPI(model, algorithm="ppo")
endpoints = NutriVisionEndpoints(api)

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    request_data = request.get_json()
    response = endpoints.handle_recommendation_request(json.dumps(request_data))
    return jsonify(json.loads(response))

@app.route('/api/session', methods=['GET'])
def get_session_summary():
    response = endpoints.handle_session_summary_request()
    return jsonify(json.loads(response))

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    response = endpoints.handle_model_info_request()
    return jsonify(json.loads(response))

@app.route('/api/export', methods=['POST'])
def export_session():
    filepath = request.json.get('filepath', 'session_log.json')
    response = endpoints.handle_export_request(filepath)
    return jsonify(json.loads(response))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
"""

def demo_api():
    """Demonstrate API usage without a trained model."""
    print("\n" + "="*80)
    print("NUTRIVISION API DEMO (Without Trained Model)")
    print("="*80 + "\n")
    
    # Create mock model for demo
    class MockModel:
        def predict(self, obs, deterministic=True):
            import random
            return random.randint(0, 3), None
    
    api = NutriVisionAPI(MockModel(), algorithm="ppo")
    
    # Example user state
    user_state = {
        "daily_calories": 1500,
        "daily_protein": 100,
        "daily_carbs": 150,
        "daily_fats": 50,
        "calorie_target": 2000,
        "protein_target": 150,
        "carbs_target": 200,
        "fats_target": 65,
        "goal_type": 0, # Weight loss
        "current_food": {
            "cal": 280,
            "protein": 6,
            "carbs": 48,
            "fat": 5
        },
        "day_progress": 0.5,
        "num_meals_logged": 3
    }
    # Get recommendation
    print("User State:")
    print(json.dumps(user_state, indent=2))
    print("\n" + "-"*80 + "\n")
    
    recommendation = api.get_recommendation(user_state)
    print("API Response:")
    print(json.dumps(recommendation, indent=2))
    
    print("\n" + "-"*80 + "\n")
    print("Session Summary:")
    print(json.dumps(api.get_session_summary(), indent=2))
    
    print("\n" + "-"*80 + "\n")
    print("Model Info:")
    print(json.dumps(api.get_model_info(), indent=2))
    
    # Export
    api.export_session_json("visualizations/api_demo_session.json")
    print("\n[OK] Session exported to visualizations/api_demo_session.json")

if __name__ == "__main__":
    demo_api()
