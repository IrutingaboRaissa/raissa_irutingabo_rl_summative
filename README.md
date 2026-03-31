# NutriVision Africa — Mission-Based Reinforcement Learning

A Gymnasium environment and training pipeline for a food-recommendation agent aligned with weight-management goals (loss, gain, maintenance), using four RL algorithms and systematic hyperparameter search.

## Overview

**Problem.** Learn a policy that chooses among accept / request alternatives / skip actions so daily intake moves toward goal-specific calorie and macro targets, using a fixed African-foods database.

**Algorithms.**

- **DQN** — value-based (Stable-Baselines3)
- **PPO** — on-policy actor–critic (SB3)
- **A2C** — advantage actor–critic (SB3)
- **REINFORCE** — vanilla policy gradient (PyTorch, custom trainer)

**Hyperparameter search.** Each algorithm defines **10** configurations (**40** runs in total when you train everything). Default budget in code is on the order of tens of thousands of timesteps per config for SB3 methods and hundreds of episodes for REINFORCE (see `train_all.py` and the trainers for exact defaults).

## Project structure

```
.
├── environment/
│   ├── custom_env.py           # NutriVisionEnv (Gymnasium)
│   ├── rendering.py            # Matplotlib / static visualization helpers
│   ├── meal_cnn.py             # MealEncoderCNN — macro “tail” for observations
│   ├── pygame_viz.py           # Pygame NutriVisionVisualizer
│   ├── api_module.py           # Optional API-related utilities
│   └── architecture_diagrams.py # System / pipeline figures (run as module)
├── training/
│   ├── dqn_training.py         # DQN, 10 HP configs → models/dqn/
│   ├── pg_training.py          # PPO + A2C + REINFORCE when run as __main__
│   └── reinforce_training.py   # REINFORCE only (can log to models/pg/)
├── models/
│   ├── dqn/                    # DQN checkpoints + dqn_summary.json (or training_summary.json)
│   └── pg/                     # PPO, A2C, REINFORCE checkpoints + *_summary.json
├── main.py                     # Playback CLI (best model from summaries)
├── play.py                     # Thin entry point: calls main.main()
├── random_demo.py              # Random policy + headless pygame screenshot demo
├── train_all.py                # Sequentially trains DQN, PPO, A2C, REINFORCE
├── analyze_results.py          # Plots and TRAINING_REPORT.txt from summaries
├── verify_project.py           # Sanity checks for key files / syntax
├── requirements.txt
└── README.md
```

## Observation space (15-D)

The environment exposes a **15-dimensional** `Box(0, 5000)` observation:

- **11 features** — hand-crafted signals (consumption vs targets, goal encoding, current meal macros, day progress, meals logged), **normalized** and scaled into the box range.
- **4 features** — outputs from **`MealEncoderCNN`** in `meal_cnn.py` (macro-related probabilities), concatenated as the tail of the vector.

If PyTorch / the CNN module is unavailable, the code can fall back without the full CNN path; see `custom_env.py` for behavior.

## Action space (discrete 4)

| Index | Action              |
|------:|---------------------|
| 0     | Accept recommendation |
| 1     | Request lower-calorie alternative |
| 2     | Request higher-protein alternative |
| 3     | Skip meal           |

Rewards favor accepting meals that fit the daily plan; requesting alternatives carries a small penalty; skip is neutral (see `custom_env.py`).

## Installation

```bash
git clone https://github.com/<your_github_username>/student_name_rl_summative.git
cd student_name_rl_summative
pip install -r requirements.txt
```

Replace `<your_github_username>` with your account. The suggested repository name for the summative is **`student_name_rl_summative`**.

## Quick start (typical workflow)

Run from the project root **after** `pip install -r requirements.txt`:

```bash
pip install -r requirements.txt
python run_local.py
```

Or run step-by-step:

```bash
pip install -r requirements.txt
python train_all.py
python analyze_results.py
python random_demo.py
python main.py --pygame
python play.py --pygame
```

- `train_all.py` trains all four algorithms (long run). Use individual scripts under `training/` if you want to train one family at a time.
- `main.py --pygame` and `play.py --pygame` are equivalent entry points for playback with the Pygame window (requires trained models and summary JSON files under `models/`).
- `run_local.py` is the single local entry point: training + analysis + videos + artifact collection under `outputs/`.

## GPU acceleration

Training uses **PyTorch** and **Stable-Baselines3**. If a **CUDA** GPU is available and your installed `torch` build supports it, the trainers pick **`cuda`** automatically for SB3 models and for the meal CNN in the environment when encoding observations.

1. Install a **GPU-enabled** PyTorch build from [pytorch.org](https://pytorch.org) (match your CUDA version).
2. Run the same commands as above; no extra flags are required for the default scripts.
3. The environment is small; the **largest speedups** are on **DQN / PPO / A2C** updates. **REINFORCE** (custom loop) also benefits on GPU if PyTorch uses CUDA.

If `torch.cuda.is_available()` is `False`, everything falls back to CPU.

## Commands

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run full local pipeline (single entry point) | `python run_local.py` |
| Train all four algorithms (full sweep) | `python train_all.py` |
| DQN only (10 configs) | `python training/dqn_training.py` |
| PPO, A2C, and REINFORCE (10 configs each) | `python training/pg_training.py` |
| REINFORCE only | `python training/reinforce_training.py` |
| Random agent demo (PNG under `visualizations/`) | `python random_demo.py` |
| Play back best agent (from JSON summaries) | `python main.py` |
| Same as `main.py` | `python play.py` |
| Playback with Pygame window | `python main.py --pygame` |
| Analysis / plots / report | `python analyze_results.py` |
| Verify layout and Python syntax | `python verify_project.py` |
| Generate architecture diagrams | `python -m environment.architecture_diagrams` |

**Playback CLI highlights** (`main.py`): `--algorithm {all,dqn,reinforce,ppo,a2c}`, `--episodes N`, `--no-render`, `--pygame`.

**Model paths.** SB3 models load from `models/dqn/` or `models/pg/` with names like `dqn_<config>`, `ppo_<config>`, `a2c_<config>`. REINFORCE weights are `models/pg/reinforce_<config>.pt`.

## Dependencies

Pinned in `requirements.txt`, including: `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `matplotlib`, `tensorboard`, `pygame`, `opencv-python`, `pillow`.

## Outputs

- **Checkpoints** under `models/dqn/` and `models/pg/`.
- **Summaries** — JSON files used by `main.py` and `analyze_results.py` (e.g. `models/dqn/dqn_summary.json`, `models/pg/ppo_summary.json`, `a2c_summary.json`, `reinforce_summary.json`).
- **`analyze_results.py`** — figures under `visualizations/` and `TRAINING_REPORT.txt`.
- **`architecture_diagrams`** — diagram images as produced by that module’s `main()`.

## Limitations and extensions

- Fixed set of **15** African dishes; portions are simulated, not from real vision.
- Suitable for coursework / experimentation rather than clinical deployment.
- Natural extensions: larger menu, real nutrition APIs, user profiles, or replacing the CNN tail with a trained image encoder.

## Author

**Raissa IRUTINGABO**

Course: Pre-Capstone — Machine Learning Specialization  

Program: Bachelor of Science in Software Engineering

## License

Provided as educational material for the Pre-Capstone course.

## References

- Stable-Baselines3 documentation  
- OpenAI Gymnasium documentation  
- African foods / nutrition literature as cited in coursework materials
