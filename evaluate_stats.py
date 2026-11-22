import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.enums import ActionType, Physics

from drone_nav.navigation_aviary import NavigationAviary


def band_of(z):
    return 'low' if z < 0.7 else 'middle' if z < 1.5 else 'high'


def classify_bars(positions, obstacles):
    out = []
    for o in obstacles:
        axis = o.get('axis', 2)
        if axis == 2:
            continue
        m = 1 - axis                      # horizontal axis perpendicular to the bar
        c = np.array(o['pos'], dtype=float)
        half = o['height'] / 2.0
        d = positions[:, m] - c[m]
        crossings = np.where(d[:-1] * d[1:] < 0)[0]
        if len(crossings) == 0:
            result, z_cross = 'not_crossed', None
        else:
            i = crossings[0]
            t = d[i] / (d[i] - d[i + 1])
            point = positions[i] + t * (positions[i + 1] - positions[i])
            z_cross = float(point[2])
            if abs(point[axis] - c[axis]) <= half:
                result = 'over' if point[2] > c[2] else 'under'
            else:
                result = 'around'
        out.append({'band': band_of(c[2]), 'axis': 'xy'[axis], 'z_bar': float(c[2]),
                    'z_cross': z_cross, 'result': result})
    return out


def run_episode(env, model, seed):
    obs, _ = env.reset(seed=seed)
    obstacles = [dict(o) for o in env.NAV_OBSTACLES]
    positions = [env._getDroneStateVector(0)[0:3].copy()]
    done, total, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        positions.append(env._getDroneStateVector(0)[0:3].copy())
        done = terminated or truncated
    positions = np.array(positions)
    if info['reached_goal']:
        outcome = 'goal'
    elif info['collided']:
        outcome = 'collision'
    elif steps / getattr(env, 'CTRL_FREQ', 30) >= env.EPISODE_LEN_SEC - 0.2:
        outcome = 'timeout'
    else:
        outcome = 'out_of_bounds_or_tilt'
    return {
        'seed': seed,
        'outcome': outcome,
        'reward': float(total),
        'steps': steps,
        'final_dist': float(info['dist_to_target']),
        'n_vertical': sum(o.get('axis', 2) == 2 for o in obstacles),
        'n_horizontal': sum(o.get('axis', 2) != 2 for o in obstacles),
        'bars': classify_bars(positions, obstacles),
    }


def summarize(rows):
    n = len(rows)
    pct = lambda k: 100.0 * k / n if n else 0.0
    oc = Counter(r['outcome'] for r in rows)
    lines = [f'episodes: {n}',
             f'reached goal : {oc["goal"]:3d}  ({pct(oc["goal"]):.0f}%)',
             f'collision    : {oc["collision"]:3d}  ({pct(oc["collision"]):.0f}%)',
             f'timeout      : {oc["timeout"]:3d}  ({pct(oc["timeout"]):.0f}%)',
             f'out of bounds / tilt: {oc["out_of_bounds_or_tilt"]:3d}  ({pct(oc["out_of_bounds_or_tilt"]):.0f}%)',
             f'mean final distance to goal: {np.mean([r["final_dist"] for r in rows]):.2f} m',
             f'mean reward: {np.mean([r["reward"] for r in rows]):.1f}']

    with_bars = [r for r in rows if r['n_horizontal'] > 0]
    without = [r for r in rows if r['n_horizontal'] == 0]
    if with_bars:
        ok = sum(r['outcome'] == 'goal' for r in with_bars)
        lines.append(f'scenes WITH horizontal bars   : {len(with_bars)}  -> goal {ok}/{len(with_bars)} '
                     f'({100.0 * ok / len(with_bars):.0f}%)')
    if without:
        ok = sum(r['outcome'] == 'goal' for r in without)
        lines.append(f'scenes WITHOUT horizontal bars: {len(without)}  -> goal {ok}/{len(without)} '
                     f'({100.0 * ok / len(without):.0f}%)')

    by_band = defaultdict(Counter)
    total = Counter()
    for r in rows:
        for b in r['bars']:
            by_band[b['band']][b['result']] += 1
            total[b['result']] += 1
    if total:
        lines.append('')
        lines.append(f'how the drone got past horizontal bars (all {sum(total.values())} bars):')
        lines.append('  band     over  under  around  not_crossed')
        for band in ('high', 'middle', 'low'):
            c = by_band[band]
            lines.append(f'  {band:<7} {c["over"]:5d} {c["under"]:6d} {c["around"]:7d} {c["not_crossed"]:12d}')
        lines.append(f'  {"total":<7} {total["over"]:5d} {total["under"]:6d} {total["around"]:7d} {total["not_crossed"]:12d}')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to a saved SB3 model (.zip)')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed_start', type=int, default=1000, help='Scenes use seeds seed_start .. seed_start+episodes-1')
    parser.add_argument('--label', type=str, default='run', help='Name used in the output file names')
    parser.add_argument('--no_horizontal', action='store_true', help='Control run: vertical obstacles only')
    parser.add_argument('--wind_std', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    kw = {'num_horizontal_range': (0, 0)} if args.no_horizontal else {}
    env = NavigationAviary(gui=False, act=ActionType.VEL, physics=Physics.PYB,
                           wind_std=args.wind_std, randomize=True, **kw)
    model = PPO.load(args.model)

    rows = []
    for i in range(args.episodes):
        rows.append(run_episode(env, model, args.seed_start + i))
        print(f'  episode {i + 1}/{args.episodes}: {rows[-1]["outcome"]}', flush=True)
    env.close()

    header = [f'model: {args.model}',
              f'label: {args.label}',
              f'horizontal bars: {"OFF (control)" if args.no_horizontal else "ON"}',
              f'seeds: {args.seed_start}..{args.seed_start + args.episodes - 1}',
              f'date: {datetime.now():%Y-%m-%d %H:%M}',
              '']
    summary = summarize(rows)
    text = '\n'.join(header + summary)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, f'eval_results_{args.label}')
    with open(stem + '.txt', 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'label': args.label, 'horizontal': not args.no_horizontal,
                   'seed_start': args.seed_start, 'summary': summary, 'episodes': rows}, f, indent=2)
    print('\n' + text)
    print(f'\nSaved {stem}.txt and {stem}.json')


if __name__ == '__main__':
    main()
