"""Run the steel-industry CO2 project pipeline from start to finish.

Usage: python main.py
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Data preprocessing", "data_preprocessing.py"),
    ("Exploratory data analysis", "eda.py"),
    ("Baseline and linear regression", "run_baselines.py"),
    ("Random Forest regression", "random_forest.py"),
    ("Gradient Boosting regression", "gradient_boosting.py"),
    ("Model evaluation and comparison", "model_evaluation.py"),
    ("Feature importance", "feature_importance.py"),
]


def main():
    for number, (description, filename) in enumerate(STEPS, start=1):
        script_path = ROOT / "src" / filename
        if not script_path.is_file():
            raise FileNotFoundError(f"Required script not found: {script_path}")

        print(f"\n[{number}/{len(STEPS)}] {description}", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            check=True,
        )
        print(f"Completed: {filename}", flush=True)

    print("\nAll steps completed successfully. Check results/ and figures/.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped: a script failed (exit code {exc.returncode}).", file=sys.stderr)
        sys.exit(exc.returncode)
    except FileNotFoundError as exc:
        print(f"\nPipeline stopped: {exc}", file=sys.stderr)
        sys.exit(1)
