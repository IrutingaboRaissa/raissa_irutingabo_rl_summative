# NutriVision Africa: RL Summative

Gymnasium environment: an agent chooses **accept / lower-cal / higher-protein / skip** on suggested meals so **daily macros** move toward **loss / gain / maintenance** targets. Four algorithms (**DQN, REINFORCE, PPO, A2C**), ten hyperparameter configs each, SB3 + custom REINFORCE trainer.


## To run the project

1. **Clone and enter the repo** (from the project root for every command below):

   ```bash
   https://github.com/IrutingaboRaissa/raissa_irutingabo_rl_summative.git
   cd raissa_irutingabo_rl_summative
   ```

2. **Python** Use **64-bit Python 3.10–3.13** (3.13 is fine; avoid the experimental *free-threaded* `python3.13t` build if NumPy errors appear).

3. **Install dependencies**:

   ```bash
   python -m pip install -r requirements.txt
   ```

   On Windows, if `python` is not on your PATH: `py -m pip install -r requirements.txt`

4. **Quick checks (no trained weights required)** — confirms the env, Pygame UI, and plotting work:

   | Command | What you should see |
   |---------|---------------------|
   | `python game_watch_demo.py` | A **Pygame window**; the UI updates with random actions. Close the window or press **Esc** to quit. |
   | `python random_demo.py` | Writes **`visualizations/random_agent_demo.png`** (or similar path printed in the terminal). |

5. **Trained agent playback**  `python play.py` and `python play.py --pygame` load **checkpoint files** under `models/` (`.zip` for SB3, `.pt` for REINFORCE). Those binaries are **gitignored** in this repo, so a **fresh clone** only has **summary JSONs** (training metrics), not the weights. To run the best agent locally, **train first** (see below), or use the machine where training was already run.

6. **Optional sanity import**:

   ```bash
   python -c "from environment.custom_env import NutriVisionEnv; e=NutriVisionEnv(); e.reset(); print('OK', e.observation_space.shape, e.action_space)"
   ```

7. **Headless / CI**  If `SDL_VIDEODRIVER=dummy` is set, Pygame may not show a real window; `game_watch_demo.py` clears `dummy` when possible so a desktop session can open a window.

## How it works

1. **`NutriVisionEnv`** (`environment/custom_env.py`) samples a **goal** and **daily targets**, then each step shows a **dish** from a fixed menu.
2. **Observation**  15-D `Box(0, 5000)`: normalized intake vs targets, goal, progress, meal signals, plus a **4-D CNN tail** (`meal_cnn.py`) over the current food (zeros if the CNN path is skipped).
3. **Reward**  On **accept**, reward reflects **macro fit** vs targets (and small goal bonuses). **Alternatives** cost **−0.5** (meal not logged). **Skip** is **0**.
4. **Terminal**  Episode length cap or **over-intake** (loss/maintenance) per env rules.
5. **Training** Scripts write checkpoints under `models/` and **summary JSONs** (mean reward per config). **`play.py`** / **`main.py`** pick the **best** config per algorithm from those summaries **and** load the matching checkpoint file.
6. **Visualization** **Pygame** UI (`environment/pygame_viz.py`). **`random_demo.py`** produces a **PNG** for a static “no trained model” demo.


## Install (detail)

```bash
git clone https://github.com/IrutingaboRaissa/raissa_irutingabo_rl_summative.git
cd raissa_irutingabo_rl_summative
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Virtual environment (recommended):**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows / Python 3.13:** If NumPy fails with `python3.13t.exe`, use the standard interpreter (`py -3.13` or `Python313\python.exe`). See [NumPy troubleshooting](https://numpy.org/devdocs/user/troubleshooting-importerror.html).

**GPU:** Optional. Install CUDA PyTorch from [pytorch.org](https://pytorch.org) if you want GPU; **CPU is enough** for demos and smaller training runs.

## Commands (cheat sheet)

| Goal | Command |
|------|---------|
| **Pygame + no trained weights** | `python game_watch_demo.py` · optional: `--delay-ms 300` |
| **Random policy → PNG** | `python random_demo.py` |
| **Best agent, terminal only** | `python play.py` *(needs checkpoints under `models/`)* |
| **Best agent + Pygame window** | `python play.py --pygame` |
| **Train everything** (long) | `python train_all.py` |
| **Train one stack** | `python training/dqn_training.py` · `python training/pg_training.py` · `python training/reinforce_training.py` |
| **Tables, plots, report text** | `python analyze_results.py` *(needs summary JSONs under `models/`)* |
| **Record `.avi` demos** | `python record_videos.py` *(needs OpenCV, checkpoints; e.g. `--headless --verbose --steps 400`)* |
| **Full local pipeline** | `python run_local.py` · add `--skip-training` if models already exist |
| **Architecture diagrams** | `python -m environment.architecture_diagrams` |

**Entry points:** `python play.py` and `python main.py` are the same CLI.

**`play.py` / `main.py` flags:** `--algorithm {all,dqn,reinforce,ppo,a2c}` · `--episodes N` · `--no-render` · `--pygame`


## Dependencies (`requirements.txt`)

| Package | Role |
|---------|------|
| `gymnasium` | Environment API |
| `stable-baselines3` | DQN, PPO, A2C |
| `torch` | Neural nets + REINFORCE |
| `pygame` | Live UI |
| `matplotlib` | Plots / static viz |
| `opencv-python` | Video export in `record_videos.py` |
| `numpy`, `pillow`, `tensorboard` | Arrays, images, optional TB logging |


## Project structure

Overview of the repository (paths are relative to the project root). Files produced by training or scripts are listed under **Generated folders**.

```
raissa_irutingabo_rl_summative/
├── README.md                 # This file
├── requirements.txt          # Pip dependencies
├── .gitignore
├── play.py                   # Entry point → delegates to main.py (same CLI as main.py)
├── main.py                   # Load best checkpoint from summaries; run episodes; optional Pygame
├── train_all.py              # Runs DQN + PG + REINFORCE training pipelines
├── analyze_results.py        # Aggregates summary JSONs → plots, tables, TRAINING_REPORT.txt
├── record_videos.py          # Records AVI demos from trained agents (OpenCV)
├── random_demo.py            # Random policy → static PNG (no trained weights)
├── game_watch_demo.py        # Pygame window + random actions (no trained weights)
├── run_local.py              # Full pipeline (train → analyze → optional extras)
├── environment/              # Gym env, UI, helpers (see below)
├── training/                 # Per-algorithm training scripts (see below)
├── models/                   # Training metrics JSONs; checkpoints after local train (see below)
└── ui_templates/             # Optional static HTML viewer (not required to run RL code)
```

### `environment/` — core RL environment and visualization

| File | Purpose |
|------|---------|
| `custom_env.py` | **`NutriVisionEnv`**: Gymnasium env (observation, actions, reward, termination), food DB, goals. |
| `pygame_viz.py` | **`NutriVisionVisualizer`**: live Pygame dashboard (client, macros, meal, rewards, transitions). |
| `rendering.py` | **`EnvironmentVisualizer`**: Matplotlib / terminal-style plots used by `main.py` and demos. |
| `meal_cnn.py` | Small CNN used to enrich observations with meal-related features (optional tail in obs). |
| `api_module.py` | Builds JSON-friendly state/recommendation payloads (report / API sketch; not required to train). |
| `architecture_diagrams.py` | Runnable module to emit architecture figures for documentation. |
| `__init__.py` | Package marker. |

### `training/` — training loops

| File | Purpose |
|------|---------|
| `dqn_training.py` | Trains **DQN** variants; writes `models/dqn/*.zip` + updates DQN summary JSON. |
| `pg_training.py` | Trains **PPO** and **A2C**; writes `models/pg/*.zip` + PG summary JSONs. |
| `reinforce_training.py` | Custom **REINFORCE** loop + `PolicyNetwork`; writes `models/pg/reinforce_*.pt` + summary. |
| `__init__.py` | Package marker. |

### `models/` — metrics and checkpoints

| Path | Purpose |
|------|---------|
| `models/dqn/dqn_summary.json` (or `training_summary.json`) | Per-config mean rewards for DQN; used to pick **best** DQN run. |
| `models/pg/ppo_summary.json` | Same for PPO. |
| `models/pg/a2c_summary.json` | Same for A2C. |
| `models/pg/reinforce_summary.json` | Same for REINFORCE. |
| `models/master_training_results.json` | Optional aggregate of runs (if present). |
| `models/dqn/*.zip`, `models/pg/*.zip` | **SB3 checkpoints** — created by training; often **not** committed (see `.gitignore`). |
| `models/pg/reinforce_*.pt` | **PyTorch** REINFORCE weights — created by training; often **not** committed. |

`play.py` reads the summary JSONs to choose the best config, then loads the matching **`.zip` / `.pt`** from disk. Without those binaries, use `game_watch_demo.py` / `random_demo.py` or train first.

### `ui_templates/`

| File | Purpose |
|------|---------|
| `index.html` | Standalone **browser UI** mockup for showcasing agent ideas; **optional** — does not replace Python/Pygame training or playback. |

### Generated folders (after running scripts; may be empty or gitignored)

| Path | What creates it | Contents |
|------|-----------------|----------|
| `visualizations/` | `random_demo.py`, `analyze_results.py`, `main.py` (static viz), etc. | PNG plots, tables, demos. |
| `videos/` | `record_videos.py` | Per-algorithm `.avi` recordings. |
| `outputs/` | `run_local.py` or custom exports | Pipeline outputs (repo-wide `outputs/` often ignored). |
| `runs/`, `logs/` | TensorBoard (if used) | Event logs. |
| `TRAINING_REPORT.txt` | `analyze_results.py` | Text summary for the report. |

## Outputs

| Path | What |
|------|------|
| `models/dqn/`, `models/pg/` | After training: checkpoints + summary JSONs |
| `visualizations/` | Plots, tables, `random_agent_demo.png`, etc. |
| `TRAINING_REPORT.txt` | From `analyze_results.py` |
| `videos/<algo>/` | From `record_videos.py` |

## API (optional)

`environment/api_module.py` builds **JSON-shaped** state/recommendation dicts for report discussion; **no cloud deploy** is required for the summative.
