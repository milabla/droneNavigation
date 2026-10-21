import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from gym_pybullet_drones.utils.enums import ActionType, Physics

from drone_nav.navigation_aviary import NavigationAviary


def make_env(wind_std: float, randomize: bool, seed: int = 0):
    def _init():
        env = NavigationAviary(gui=False,
                                act=ActionType.VEL,
                                physics=Physics.PYB,
                                wind_std=wind_std,
                                randomize=randomize)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=300_000,
                         help='Total number of environment steps to train for.')
    parser.add_argument('--n_envs', type=int, default=4,
                         help='Number of parallel environments (use ~ number of CPU cores).')
    parser.add_argument('--wind_std', type=float, default=0.0,
                         help='Std-dev of random wind force (N). 0 = deterministic dynamics.')
    parser.add_argument('--fixed_scene', action='store_true',
                         help='Disable domain randomization: train/evaluate on one fixed '
                              'target+obstacle layout only (memorization baseline).')
    parser.add_argument('--logdir', type=str, default='./logs',
                         help='Directory for TensorBoard logs and checkpoints.')
    parser.add_argument('--resume', type=str, default=None,
                         help='Path to a saved model (.zip) to resume training from.')
    args = parser.parse_args()
    randomize = not args.fixed_scene

    os.makedirs(args.logdir, exist_ok=True)

    # Vectorized training environments (run in parallel subprocesses)
    train_env = SubprocVecEnv([make_env(args.wind_std, randomize, seed=i) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    # Separate evaluation environment (single, deterministic-seeded)
    eval_env = SubprocVecEnv([make_env(args.wind_std, randomize, seed=1000)])
    eval_env = VecMonitor(eval_env)

    if args.resume is not None:
        print(f'Resuming training from {args.resume}')
        model = PPO.load(args.resume, env=train_env, tensorboard_log=args.logdir)
    else:
        model = PPO(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=3e-4,
            n_steps=1024,          # per-environment rollout length before each update
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,        # small entropy bonus helps exploration
            verbose=1,
            tensorboard_log=args.logdir,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, 'best_model'),
        log_path=os.path.join(args.logdir, 'eval'),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10 if randomize else 5,  # more eval episodes needed to average over random scenes
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(args.logdir, 'checkpoints'),
        name_prefix='ppo_drone_nav',
    )

    print(f'Domain randomization: {"on (new scene every episode)" if randomize else "off (fixed scene)"}')

    model.learn(total_timesteps=args.timesteps,
                callback=[eval_callback, checkpoint_callback],
                progress_bar=True)

    final_path = os.path.join(args.logdir, 'final_model')
    model.save(final_path)
    print(f'Training finished. Final model saved to {final_path}.zip')
    print(f'Best model (by eval reward) saved to {args.logdir}/best_model/best_model.zip')
    print(f'View training curves with: tensorboard --logdir {args.logdir}')

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
