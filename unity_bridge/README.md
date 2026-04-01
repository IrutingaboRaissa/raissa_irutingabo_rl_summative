# Unity Bridge (Replay + Interactive)

This folder helps you visualize NutriVision in Unity in two ways:
- **Replay mode**: play exported Python trajectories.
- **Interactive mode**: run the environment logic inside Unity and choose actions with keyboard input.

## 1) Export trajectories from Python

Run from project root:

```bash
python unity_export.py --algorithm ppo --episodes 3 --out outputs/unity/replay_trajectories.json
```

This writes a Unity-ready JSON file with:
- episode metadata (goal, targets, total reward),
- per-step action, reward, food name, daily stats,
- observation vectors.

## 2) Unity setup

1. Create a Unity project (3D URP recommended).
2. Add a scene with at least one empty GameObject for the controller.
   - If `autoCreateDemoScene` is enabled (default), the script auto-creates:
     - a ground plane,
     - an avatar capsule,
     - TMP UI texts,
     - calories/protein progress bars.
3. Copy these files into `Assets/Scripts/`:
   - `NutriVisionReplayController.cs`
   - `NutriVisionUnityEnv.cs`
4. (Replay mode only) Put `replay_trajectories.json` in `Assets/StreamingAssets/`.
5. Attach `NutriVisionReplayController` to an empty GameObject.
   - You can leave references empty for quick preview; they will be auto-wired.
6. Press Play.

## 3) Controls

### Replay mode
- `Space`: Pause/resume
- `N`: Jump to next episode
- `R`: Restart replay

### Interactive mode
Set `Run Mode` to `InteractiveUnityEnv` in inspector, then:
- `1`: Accept
- `2`: Lower Calorie
- `3`: Higher Protein
- `4`: Skip
- `R`: Reset episode

## 4) Notes

- Interactive mode runs Unity-side environment logic for demos.
- Replay mode uses trained policies from Python exports.
- It keeps your Python training stack unchanged.
- For capstone, you can later migrate to Unity ML-Agents while preserving the same action/reward logic.

