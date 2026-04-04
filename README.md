# NutriVision Africa — RL Summative

Gymnasium environment: an agent chooses **accept / lower-cal / higher-protein / skip** on suggested meals so **daily macros** move toward **loss / gain / maintenance** targets. Four algorithms (**DQN, REINFORCE, PPO, A2C**), ten hyperparameter configs each, SB3 + custom REINFORCE trainer.

---

## How it works

1. **`NutriVisionEnv`** (`environment/custom_env.py`) samples a **goal** and **daily targets**, then each step shows a **dish** from a fixed menu.
2. **Observation** — 15-D `Box(0, 5000)`: normalized intake vs targets, goal, progress, meal signals, plus a **4-D CNN tail** (`meal_cnn.py`) over the current food (zeros if CNN path skipped).
3. **Reward** — On **accept**, reward reflects **macro fit** vs targets (and small goal bonuses). **Alternatives** cost **−0.5** (meal not logged). **Skip** is **0**.
4. **Terminal** — Episode length cap or **over-intake** (loss/maintenance) per env rules.
5. **Training** — Each script writes checkpoints under `models/` and a **summary JSON** (mean reward per config). **`play.py`** loads the **best** config per algorithm from those summaries.
6. **Visualization** — **Pygame** UI (`environment/pygame_viz.py`): client panel, intake bars, suggested meal, rewards. **`random_demo.py`** = random policy PNG (assignment “no model” static demo).

---

## Install

```bash
git clone https://github.com/<you>/student_name_rl_summative.git
cd student_name_rl_summative
pip install -r requirements.txt
```

**Windows / Python 3.13:** if NumPy fails with `python3.13t.exe`, use the normal interpreter (`py -3.13` or `Python313\python.exe`). See [NumPy troubleshooting](https://numpy.org/devdocs/user/troubleshooting-importerror.html).

---

## Run (cheat sheet)

| Goal | Command |
|------|---------|
| **Pygame + no trained weights** (watch UI move) | `python game_watch_demo.py` |
| **Random policy → PNG** (rubric: static viz, no model) | `python random_demo.py` |
| **Best agent, terminal only** | `python play.py` |
| **Best agent, Pygame + verbose terminal** (markers / video) | `python play.py --pygame` |
| **Train everything** (long) | `python train_all.py` |
| **Train one stack** | `python training/dqn_training.py` · `python training/pg_training.py` · `python training/reinforce_training.py` |
| **Tables, plots, `TRAINING_REPORT.txt`** | `python analyze_results.py` (needs summary JSONs under `models/`) |
| **Save `.avi` demos** (needs checkpoints) | `python record_videos.py --headless --verbose --steps 400` |
| **Full pipeline + `outputs/`** | `python run_local.py` (add `--skip-training` if models exist) |
| **Diagrams for the report** | `python -m environment.architecture_diagrams` |

**`play.py` flags:** `--algorithm {all,dqn,reinforce,ppo,a2c}` · `--episodes N` · `--no-render` · `--pygame`

**GPU:** Install CUDA PyTorch from [pytorch.org](https://pytorch.org) if you want GPU; otherwise CPU is fine.

---

## Project layout (short)

```
environment/   custom_env.py, pygame_viz.py, rendering.py, meal_cnn.py, api_module.py, architecture_diagrams.py
training/      dqn_training.py, pg_training.py, reinforce_training.py
models/dqn/    DQN .zip + dqn_summary.json (or training_summary.json)
models/pg/     PPO, A2C, REINFORCE + *_summary.json
main.py, play.py          playback (same CLI)
train_all.py, analyze_results.py, record_videos.py, random_demo.py, game_watch_demo.py, run_local.py
```

---

## Outputs

| Path | What |
|------|------|
| `models/dqn/`, `models/pg/` | Checkpoints + summary JSONs |
| `visualizations/` | HP tables PNGs, comparison plots, `random_agent_demo.png`, etc. |
| `TRAINING_REPORT.txt` | From `analyze_results.py` |
| `videos/<algo>/` | From `record_videos.py` |

---

## API / “production” (optional, no deploy required)

`environment/api_module.py` **serializes env state + recommendation to JSON-shaped dicts** — useful in the **report** to show how a backend could feed a web/mobile client. You do **not** need Render, Flask, or a public URL for the summative; markers care that the **idea** is clear and the repo **runs locally**.

---

## Submission reminders

- **PDF report (7–10 pages):** map env (actions, obs, reward, termination), four HP tables / discussion, diagrams, and any **training curves** (TensorBoard screenshots if you add logging).
- **Video:** fullscreen, camera on, state problem / objective / rewards, show **Pygame + terminal**, run **best** agent, narrate behavior.
- **Clone test:** fresh `git clone` → `pip install -r requirements.txt` → at least `python random_demo.py` and `python game_watch_demo.py` should work without training.

---

## License & references

Educational use (Pre-Capstone). See Stable-Baselines3, Gymnasium, and your course readings for citations.
