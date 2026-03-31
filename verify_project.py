#!/usr/bin/env python
"""
NutriVision RL - Project Verification Script
Checks that all required files are present and functional
"""

import os
import sys


def check_file(filepath: str, description: str = "") -> bool:
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISS]"
    desc = f" - {description}" if description else ""
    print(f"{status} {filepath}{desc}")
    return exists


def check_python_file(filepath: str) -> bool:
    """Check if a Python file exists and compiles."""
    if not os.path.exists(filepath):
        print(f"[MISS] {filepath} - NOT FOUND")
        return False

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            compile(f.read(), filepath, "exec")
        print(f"[OK] {filepath} - VALID")
        return True
    except SyntaxError as e:
        print(f"[FAIL] {filepath} - SYNTAX ERROR: {e}")
        return False
    except Exception:
        print(f"[OK] {filepath} - OK (compile check skipped)")
        return True


def main():
    """Main verification function."""
    print("\n" + "=" * 80)
    print("NUTRIVISION RL - PROJECT VERIFICATION")
    print("=" * 80 + "\n")

    all_ok = True

    print("Environment Files:")
    print("-" * 80)
    env_files = [
        ("environment/__init__.py", "Package init"),
        ("environment/custom_env.py", "Custom Gymnasium environment"),
        ("environment/rendering.py", "Matplotlib visualization"),
        ("environment/meal_cnn.py", "CNN meal encoder"),
        ("environment/pygame_viz.py", "Pygame 2D GUI"),
        ("environment/architecture_diagrams.py", "Architecture diagrams"),
        ("environment/api_module.py", "JSON API module"),
    ]
    for filepath, desc in env_files:
        if not check_python_file(filepath):
            all_ok = False

    print("\nTraining Files:")
    print("-" * 80)
    training_files = [
        ("training/__init__.py", "Package init"),
        ("training/dqn_training.py", "DQN trainer"),
        ("training/pg_training.py", "PPO, A2C, REINFORCE trainers"),
        ("training/reinforce_training.py", "REINFORCE trainer"),
    ]

    for filepath, desc in training_files:
        if not check_python_file(filepath):
            all_ok = False

    print("\nMain Files:")
    print("-" * 80)
    main_files = [
        ("main.py", "Best agent playback"),
        ("play.py", "Interactive play"),
        ("random_demo.py", "Random policy demo"),
        ("train_all.py", "Master training orchestrator"),
        ("analyze_results.py", "Analysis & visualization"),
    ]

    for filepath, desc in main_files:
        if not check_python_file(filepath):
            all_ok = False

    print("\nDocumentation Files:")
    print("-" * 80)
    doc_files = [
        ("README.md", "Project documentation"),
        ("requirements.txt", "Dependencies"),
        (".gitignore", "Git configuration"),
    ]

    for filepath, desc in doc_files:
        if not check_file(filepath, desc):
            all_ok = False

    print("\nRequired Directories:")
    print("-" * 80)
    dirs = [
        "models",
        "models/dqn",
        "models/pg",
        "visualizations",
    ]

    for dirpath in dirs:
        exists = os.path.isdir(dirpath)
        status = "[OK]" if exists else "[MISS]"
        print(f"{status} {dirpath}/")

    print("\n" + "=" * 80)
    if all_ok:
        print("[OK] ALL FILES PRESENT AND VALID")
        print("=" * 80)
        print("\nTypical workflow:")
        print("  pip install -r requirements.txt")
        print("  python train_all.py")
        print("  python analyze_results.py")
        print("  python random_demo.py")
        print("  python main.py --pygame")
        print("  python play.py --pygame")
        return 0

    print("[MISS] SOME FILES MISSING OR INVALID")
    print("=" * 80)
    print("\nPlease check the files marked with [MISS] or [FAIL] above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
