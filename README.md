# Drone Navigation with Deep Reinforcement Learning



## Project overview

A simulated quadcopter (Crazyflie 2.X) learns to fly on its own from a start
point to a goal in 3D space while avoiding obstacles. Nobody writes the flight
rules by hand: the drone learns by trial and error in a physics simulation
(**PyBullet**, through the `gym-pybullet-drones` library), using the **PPO**
algorithm from `stable-baselines3`.

- **Task:** the drone starts at (0, 0, 1) m and has to reach a randomly placed goal.
- **Obstacles:** vertical cylinders (pillars) that are flown around or over,
  and horizontal cylinders (poles) at three heights (low, middle, high), where
  the drone decides whether to pass over, under or around.
- **Reward:** positive when the drone gets closer to the goal and when it
  reaches it, negative when it crashes.
- **Generalization:** every new attempt uses a new random scene (domain
  randomization), so the drone learns a general rule instead of one path.
- **Result:** on 1000 fixed scenes with horizontal obstacles, the model trained
  with horizontal obstacles reaches the goal in 67.5% of the episodes (versus
  60.9% for the model trained only with vertical obstacles).


## Files

- `drone_nav/navigation_aviary.py` - the environment: places the goal and the
  obstacles, defines what the drone sees (observation), the reward and when an
  episode ends.
- `drone_nav/__init__.py` - makes the `drone_nav` folder a Python package.
- `train.py` - trains the PPO model and saves the models (checkpoints,
  `best_model`, `final_model`).
- `evaluate.py` - loads a saved model, flies a few episodes and draws 3D
  trajectory plots (or shows a live flight with `--gui`).
- `evaluate_stats.py` - numeric evaluation on fixed scenes (seed 1000
  onwards): how many episodes reached the goal, collided or timed out, and
  how the drone passes the horizontal obstacles (over/under/around). Writes a
  `.txt` and a `.json` file.
- `requirements.txt` - list of packages and versions.
- `eval_after_horizontal.png`, `my_eval_updated.png` - example plots from
  evaluating a model (`evaluate.py --out ...`).
- `proba.py` - a small check that the packages are installed and working.
- `.gitignore` - files that Git ignores.

## Installation

```bash
# 1. gym-pybullet-drones (must be installed from source)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
cd ..

# 2. the remaining packages
pip install -r requirements.txt

# if you get "No module named 'pkg_resources'":
pip install "setuptools<81"
```

Use the Python interpreter of the project (in PyCharm, open the Terminal
inside the project, which automatically uses the right interpreter).

Check that everything is installed:

```bash
python proba.py
```

## How to run the commands

Run all commands from the main folder of the project.

### Training (`train.py`)

```bash
# Quick test (a few minutes)
python train.py --timesteps 5000 --n_envs 2

# Train from scratch with the curriculum (from easy to hard)
python train.py --timesteps 1000000

# Continue from a saved model, at full difficulty, without the curriculum
python train.py --resume logs/final_model.zip --no_curriculum --timesteps 1500000 --logdir ./logs_horizontal

# Fixed scene (only a "memorization" baseline for comparison)
python train.py --timesteps 300000 --fixed_scene --logdir ./logs_fixed

# With random wind
python train.py --timesteps 1000000 --wind_std 0.15 --logdir ./logs_wind

# Watch the learning curves
python -m tensorboard.main --logdir ./logs
```

Options: `--timesteps` (total steps), `--n_envs` (parallel environments),
`--logdir` (output folder), `--resume` (model to continue from),
`--no_curriculum` (full difficulty from the start), `--fixed_scene`,
`--wind_std`.

Output: `logdir/checkpoints/` (every 50,000 steps),
`logdir/best_model/best_model.zip` and `logdir/final_model.zip`.

Horizontal obstacles are enabled by default. They are disabled with
`num_horizontal_range=(0, 0)` in the environment.

### Showing flights (`evaluate.py`)

```bash
# Trajectory plots (one subplot per episode)
python evaluate.py --model logs/best_model/best_model.zip --episodes 4

# Save them as an image
python evaluate.py --model logs/best_model/best_model.zip --episodes 4 --out flights.png

# Live flight on screen (PyBullet window)
python evaluate.py --model logs/best_model/best_model.zip --episodes 3 --gui
```

Options: `--model` (required), `--episodes`, `--out`, `--gui`,
`--fixed_scene`, `--wind_std`.

### Numeric evaluation (`evaluate_stats.py`)

```bash
# 1000 fixed scenes (seeds 1000-1999) with horizontal obstacles
python evaluate_stats.py --model logs/final_model.zip --episodes 1000 --label my_model

# The same, but with vertical obstacles only
python evaluate_stats.py --model logs/final_model.zip --episodes 1000 --label my_model_vertical --no_horizontal
```

Options: `--model` (required), `--episodes` (default 50), `--seed_start`
(default 1000), `--label` (name of the output files), `--no_horizontal`,
`--wind_std`, `--outdir`.

Output: `eval_results_<label>.txt` and `eval_results_<label>.json`.
An evaluation of 1000 scenes takes about 40 minutes. Using the same `--label`
overwrites the old results.
