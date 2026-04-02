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
├── main.py / play.py           # Playback CLI (best model from summaries); play.py is the thin entry
├── random_demo.py              # Random policy + static visualization demo
├── record_videos.py            # Record per-algorithm .avi demos under videos/<algo>/
├── unity_export.py             # Export trajectories for Unity replay
├── prepare_unity_demo.ps1      # Windows: export JSON (optional copy to Unity project)
├── train_all.py                # Sequentially trains DQN, PPO, A2C, REINFORCE
├── run_local.py                # Optional full pipeline + outputs/ collector
├── analyze_results.py          # Plots and TRAINING_REPORT.txt from summaries
├── unity_bridge/               # Unity replay script + integration notes
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

**Windows / Python 3.13.** If `import numpy` fails and your error mentions `python3.13t.exe`, the launcher is using the **free-threaded** build. Use the standard interpreter, e.g. `py -3.13` instead of `py`, or `Python313\python.exe` (see [NumPy troubleshooting](https://numpy.org/devdocs/user/troubleshooting-importerror.html)).

---

## How to run the project (what and why)

Run everything from the **project root** after `pip install -r requirements.txt`. `play.py` is the same entry as `main.py` (thin wrapper).

| What you want | Command | Why |
|----------------|---------|-----|
| **Install dependencies** | `pip install -r requirements.txt` | Gymnasium, SB3, PyTorch, Pygame, OpenCV, etc. |
| **Train everything (long)** | `python train_all.py` | Runs all four algorithms and hyperparameter configs; writes **checkpoints** under `models/dqn/` and `models/pg/` plus **summary JSONs** used for playback and analysis. |
| **Train one family only** | `python training/dqn_training.py` or `python training/pg_training.py` or `python training/reinforce_training.py` | Same idea as above but **DQN only**, **PPO+A2C+REINFORCE**, or **REINFORCE only** — faster iteration. |
| **Plots, tables, written report** | `python analyze_results.py` | Reads summary JSONs → figures under `visualizations/`, `TRAINING_REPORT.txt`, comparison plots for the write-up. |
| **Live demo: GUI + verbose terminal** | `python play.py --pygame` or `python main.py --pygame` | **Best policy** from summaries steps the env; **Pygame** shows the sim, **terminal** prints each step (food, action, reward, macros). Use for **markers** or **screen-recorded video demos**. Add `--episodes N`, `--algorithm ppo` (etc.). |
| **Playback, terminal only** | `python play.py` | Same agent, **no Pygame**; prints step logs. Use `--no-render` to silence terminal. |
| **Random policy screenshot** | `python random_demo.py` | Quick **baseline** visualization (PNG under `visualizations/`); no trained weights. |
| **Save demo videos per algorithm** | `python record_videos.py --headless --steps 400` | Writes **`.avi` (MJPEG)** under `videos/dqn/`, `videos/ppo/`, etc., with **labeled filenames** (algorithm + best config + step cap). Needs **trained checkpoints**. Add `--algorithm ppo` for one algo, `--timestamp` to avoid overwrites, `--verbose` for step prints. |
| **Full pipeline in one go** | `python run_local.py` | **Train → analyze → random demo PNG → copy artifacts to `outputs/`**. Add `--skip-training` if models already exist; add `--with-videos --headless` to also run `record_videos.py` into `videos/`. |
| **Unity replay JSON** | `python unity_export.py --algorithm ppo --episodes 3` | Exports trajectories for **Unity**; needs checkpoints. Use `--random-policy` for JSON **without** weights (smoke test). |
| **Architecture diagrams** | `python -m environment.architecture_diagrams` | Generates pipeline / system figures if you need them for documentation. |

### Typical flows

**A — From scratch (reproduce full project)**  
`pip install -r requirements.txt` → `python train_all.py` → `python analyze_results.py` → `python play.py --pygame` (or `record_videos.py` for files to submit).

**B — You already have `models/` checkpoints + summaries**  
`python analyze_results.py` (if you need fresh plots) → `python play.py --pygame` for live demo, or `python record_videos.py --headless` for saved videos.

**C — Automated packaging**  
`python run_local.py` (optionally `--skip-training`, `--with-videos --headless`). Collects models, visualizations, tables, and videos into `outputs/`.

### Playback details

- **Flags** (`main.py` / `play.py`): `--algorithm {all,dqn,reinforce,ppo,a2c}`, `--episodes N`, `--no-render`, `--pygame`.
- **Checkpoints:** SB3 loads `models/dqn/dqn_<config>` or `models/pg/{ppo,a2c}_<config>` (`.zip` if present). REINFORCE: `models/pg/reinforce_<config>.pt`. The **best** config per algorithm is the one with **highest mean reward** in that algorithm’s summary JSON.

## GPU acceleration

Training uses **PyTorch** and **Stable-Baselines3**. If a **CUDA** GPU is available and your installed `torch` build supports it, the trainers pick **`cuda`** automatically for SB3 models and for the meal CNN in the environment when encoding observations.

1. Install a **GPU-enabled** PyTorch build from [pytorch.org](https://pytorch.org) (match your CUDA version).
2. Run the same commands as above; no extra flags are required for the default scripts.
3. The environment is small; the **largest speedups** are on **DQN / PPO / A2C** updates. **REINFORCE** (custom loop) also benefits on GPU if PyTorch uses CUDA.

If `torch.cuda.is_available()` is `False`, everything falls back to CPU.

## Dependencies

Pinned in `requirements.txt`, including: `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `matplotlib`, `tensorboard`, `pygame`, `opencv-python`, `pillow`.

## Outputs

- **Checkpoints** under `models/dqn/` and `models/pg/`.
- **Summaries** — JSON files used by `main.py` and `analyze_results.py` (e.g. `models/dqn/dqn_summary.json`, `models/pg/ppo_summary.json`, `a2c_summary.json`, `reinforce_summary.json`).
- **`analyze_results.py`** — figures under `visualizations/` and `TRAINING_REPORT.txt`.
- **`record_videos.py`** — labeled **`.avi`** files under **`videos/<algorithm>/`** (see table above).
- **`architecture_diagrams`** — diagram images as produced by that module’s `main()`.
- **`unity_export.py`** — replay JSON (default `outputs/unity/` or path you pass) for Unity visualization.

## Unity integration (replay mode)

You can keep Python for RL training and use Unity for polished visualization:

1. Export replay data (trained policy — needs checkpoints under `models/`):
   ```bash
   python unity_export.py --algorithm ppo --episodes 3 --out outputs/unity/replay_trajectories.json
   ```
   If you only need a **Unity replay smoke test** and have no `.zip` / `.pt` weights yet:
   ```bash
   python unity_export.py --random-policy --episodes 3 --out outputs/unity/replay_trajectories.json
   ```
   On Windows you can also run `.\prepare_unity_demo.ps1` (default: random policy, uses `py -3.13`). Use `-Trained` after you have checkpoints; add `-UnityProjectPath "C:\path\to\YourUnityProject"` to copy JSON into `Assets/StreamingAssets/`.
2. Install **Unity Hub** + an LTS Editor, create a **3D (URP)** project, import **TextMeshPro** when prompted.
3. Follow `unity_bridge/README.md`: copy the two `.cs` scripts into `Assets/Scripts/`, put `replay_trajectories.json` in `Assets/StreamingAssets/`, attach `NutriVisionReplayController` to a GameObject, press **Play**.

The Unity bridge also includes an **interactive mode** (inspector: `Run Mode` → `InteractiveUnityEnv`) where you press `1/2/3/4` for actions with **no JSON file** — useful if you only want a live demo inside Unity.

## Rubric checklist (how to hit the marks)

**Environment validity** — `environment/custom_env.py` defines a 15-D observation, 4 discrete actions (accept / lower-cal / higher protein / skip), goal-dependent targets (loss / gain / maintenance), rewards tied to macro alignment and small penalties for alternatives, termination at `max_steps` or extreme over-intake. For the write-up, explicitly map each action to state changes and name edge cases (skip, alternatives, target blowout).

**Hyperparameter tables** — After training, run `python analyze_results.py`. It writes four PNG tables under `visualizations/` (`dqn_results_table.png`, `ppo_results_table.png`, `a2c_results_table.png`, `reinforce_results_table.png`) with **10 rows** and distinct hyperparameter columns. In your report, interpret **lr, gamma, batch/n_steps, exploration (DQN), entropy/clip (PPO)** with reference to stability vs convergence.

**Visualization / agent behavior** — Run `python main.py --pygame` or `python play.py --pygame` so the marker sees **Pygame GUI + terminal** together. `environment/api_module.py` is your hook for “JSON / frontend” narrative (serialize state + recommendation).

**Discussion graphs** — You already get hp bar charts, algorithm comparison, generalization plot, and tables from `analyze_results.py`. The rubric also names **cumulative reward curves, DQN loss/objective curves, PG entropy curves, convergence plots**: those need **training-time logs** (e.g. SB3 `TensorboardCallback` or CSV from callbacks). Easiest path: add TensorBoard logging per algorithm, train once, screenshot the relevant scalars for the report.

**Video demo (5 pts)** — Record **full screen**, **camera on you**, state the **problem, objective, reward rules**, show **simulation (Pygame) + verbose terminal** side by side, and narrate **what the agent does each step**. Prefer a **longer run** so the file is not trivial:
`python record_videos.py --headless --verbose --steps 400`  
Outputs are **`.avi` (MJPEG)** under **`videos/dqn/`**, **`videos/ppo/`**, **`videos/a2c/`**, **`videos/reinforce/`**, with filenames like `NutriVision_DQN_best_large_buffer_max400steps.avi`. Add `--timestamp` to avoid overwrites. Use **VLC** if the default player fails; re-encode with Clipchamp / OBS if your LMS requires MP4.

## Limitations and extensions

- Fixed set of **15** African dishes; portions are simulated, not from real vision.
- Suitable for coursework / experimentation rather than clinical deployment.
- Natural extensions: larger menu, real nutrition APIs, user profiles, or replacing the CNN tail with a trained image encoder.


## License

Provided as educational material for the Pre-Capstone course.

## References

- Stable-Baselines3 documentation  
- OpenAI Gymnasium documentation  
- African foods / nutrition literature as cited in coursework materials
