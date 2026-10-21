import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from stable_baselines3 import PPO
from gym_pybullet_drones.utils.enums import ActionType, Physics

from drone_nav.navigation_aviary import NavigationAviary


def plot_cylinder(ax, center, radius, height, color, alpha=0.35):
    """Draws a vertical cylinder obstacle centered at center
    (base is at center_z - height/2, top at center_z + height/2)."""
    z_base = center[2] - height / 2.0
    theta = np.linspace(0, 2 * np.pi, 24)
    z = np.linspace(z_base, z_base + height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)
    # simple caps
    for zc in (z_base, z_base + height):
        cx = center[0] + radius * np.cos(theta)
        cy = center[1] + radius * np.sin(theta)
        ax.plot(cx, cy, zc, color=color, alpha=alpha)


def plot_sphere(ax, center, radius, color, alpha=0.5):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def run_episode(env, model, deterministic=True):
    obs, info = env.reset()
    # snapshot the scene for this episode (it may differ from the next one if domain randomization is on)
    obstacles = [dict(o) for o in env.NAV_OBSTACLES]
    target = env.TARGET_POS.copy()

    positions = []
    done = False
    total_reward = 0.0
    while not done:
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    positions.append(env._getDroneStateVector(0)[0:3].copy())
    return np.array(positions), total_reward, info, obstacles, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to a saved SB3 model (.zip)')
    parser.add_argument('--episodes', type=int, default=4, help='Number of episodes to run and plot')
    parser.add_argument('--wind_std', type=float, default=0.0, help='Wind std-dev (should match training)')
    parser.add_argument('--fixed_scene', action='store_true',
                         help='Disable domain randomization: reuse the one fixed layout every episode.')
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI (only works with a display)')
    parser.add_argument('--out', type=str, default='trajectory.png', help='Output plot filename')
    args = parser.parse_args()

    env = NavigationAviary(gui=args.gui, act=ActionType.VEL, physics=Physics.PYB,
                            wind_std=args.wind_std, randomize=not args.fixed_scene)
    model = PPO.load(args.model)

    n_cols = min(args.episodes, 2)
    n_rows = math.ceil(args.episodes / n_cols)
    fig = plt.figure(figsize=(7 * n_cols, 6.5 * n_rows))

    for ep in range(args.episodes):
        positions, total_reward, info, obstacles, target = run_episode(env, model)

        ax = fig.add_subplot(n_rows, n_cols, ep + 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color='tab:blue', linewidth=2)
        ax.scatter(*positions[0], color='black', marker='o', s=60, label='start')
        ax.scatter(*positions[-1], color='tab:blue', marker='X', s=90, label='end')

        for obst in obstacles:
            plot_cylinder(ax, obst['pos'], obst['radius'], obst['height'], color='red')

        plot_sphere(ax, target, env.GOAL_RADIUS, color='green')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z / height (m)')
        ax.set_title(f'Episode {ep + 1}  R={total_reward:.1f}  '
                     f'goal={info["reached_goal"]}  collision={info["collided"]}', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved trajectory plot to {args.out}')

    env.close()


if __name__ == '__main__':
    main()
