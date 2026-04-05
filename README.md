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


## Project layout (short)

```
environment/   custom_env.py, pygame_viz.py, rendering.py, meal_cnn.py, api_module.py, architecture_diagrams.py
training/      dqn_training.py, pg_training.py, reinforce_training.py
models/dqn/    summary JSONs + DQN .zip (after training; .zip gitignored by default)
models/pg/     summary JSONs + PPO, A2C, REINFORCE files (after training)
main.py, play.py          same playback CLI
train_all.py, analyze_results.py, record_videos.py, random_demo.py, game_watch_demo.py, run_local.py
```

## Outputs

| Path | What |
|------|------|
| `models/dqn/`, `models/pg/` | After training: checkpoints + summary JSONs |
| `visualizations/` | Plots, tables, `random_agent_demo.png`, etc. |
| `TRAINING_REPORT.txt` | From `analyze_results.py` |
| `videos/<algo>/` | From `record_videos.py` |

## API (optional)

`environment/api_module.py` builds **JSON-shaped** state/recommendation dicts for report discussion; **no cloud deploy** is required for the summative.
