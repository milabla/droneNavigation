import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from gym_pybullet_drones.utils.enums import ActionType, Physics

from drone_nav.navigation_aviary_v5 import NavigationAviary


def make_env(wind_std: float, randomize: bool, seed: int = 0, difficulty: float = 1.0):
    def _init():
        env = NavigationAviary(gui=False,
                                act=ActionType.VEL,
                                physics=Physics.PYB,
                                wind_std=wind_std,
                                randomize=randomize,
                                difficulty=difficulty)
        env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class CurriculumCallback(BaseCallback):
    # Raises difficulty by step whenever the recent training success rate exceeds threshold

    def __init__(self, start=0.0, step=0.1, threshold=0.5, min_episodes=50, verbose=1):
        super().__init__(verbose)
        self.difficulty = start
        self.step_size = step
        self.threshold = threshold
        self.min_episodes = min_episodes

    def _on_training_start(self):
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_success_buffer
        self.logger.record("curriculum/difficulty", self.difficulty)
        if self.difficulty < 1.0 and len(buf) >= self.min_episodes and float(np.mean(buf)) >= self.threshold:
            self.difficulty = min(1.0, round(self.difficulty + self.step_size, 3))
            self.training_env.env_method("set_difficulty", self.difficulty)
            buf.clear()   # remeasure success at the new difficulty
            if self.verbose:
                print(f'[curriculum] difficulty -> {self.difficulty:.2f} at step {self.num_timesteps}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--wind_std', type=float, default=0.0)
    parser.add_argument('--fixed_scene', action='store_true')
    parser.add_argument('--no_curriculum', action='store_true', help='Train at full difficulty from the start.')
    parser.add_argument('--logdir', type=str, default='./logs_v5')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    randomize = not args.fixed_scene

    os.makedirs(args.logdir, exist_ok=True)

    train_env = SubprocVecEnv([make_env(args.wind_std, randomize, seed=i) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99)

    eval_env = SubprocVecEnv([make_env(args.wind_std, randomize, seed=1000)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)

    if args.resume is not None:
        print(f'Resuming training from {args.resume}')
        model = PPO.load(args.resume, env=train_env, tensorboard_log=args.logdir)
    else:
        model = PPO(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=linear_schedule(3e-4),
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log=args.logdir,
        )

    callbacks = [
        EvalCallback(eval_env,
                     best_model_save_path=os.path.join(args.logdir, 'best_model'),
                     log_path=os.path.join(args.logdir, 'eval'),
                     eval_freq=max(10_000 // args.n_envs, 1),
                     n_eval_episodes=10 if randomize else 5,
                     deterministic=True,
                     render=False),
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=os.path.join(args.logdir, 'checkpoints'),
                           name_prefix='ppo_drone_nav_v5'),
    ]
    if randomize and not args.no_curriculum:
        callbacks.append(CurriculumCallback())

    print(f'Domain randomization: {"on" if randomize else "off"}; '
          f'curriculum: {"on" if (randomize and not args.no_curriculum) else "off"}')

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(args.logdir, 'final_model')
    model.save(final_path)
    print(f'Training finished. Final model saved to {final_path}.zip')
    print(f'Best model (by eval reward at full difficulty): {args.logdir}/best_model/best_model.zip')

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
