import argparse
import os.path as osp
from gym.wrappers import Monitor
import numpy as np

from common import logger
from common.atari_wrappers import make_atari, wrap_deepmind
import dqfd


def train(args):
    total_timesteps = int(args.num_timesteps)
    pre_train_timesteps = int(args.pre_train_timesteps)
    seed = args.seed

    env = make_env(args.env, args.seed, args.max_episode_steps, wrapper_kwargs={'frame_stack': True})
    if args.save_video_interval != 0:
        env = Monitor(env, osp.join(logger.get_dir(), "videos"), video_callable=(lambda ep: ep % 1 == 0), force=True)
    model = dqfd.learn(
        env=env,
        network='cnn',
        seed=seed,
        total_timesteps=total_timesteps,
        pre_train_timesteps=pre_train_timesteps,
        load_path=args.load_path,
        demo_path=args.demo_path,
    )

    return model, env


def make_env(env_id, seed=None, max_episode_steps=None, wrapper_kwargs=None):
    wrapper_kwargs = wrapper_kwargs or {}
    env = make_atari(env_id, max_episode_steps)
    env.seed(seed)
    env = wrap_deepmind(env, **wrapper_kwargs)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str, default='atari')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_timesteps', help='', type=float, default=1e6)
    parser.add_argument('--pre_train_timesteps', help='', type=float, default=750000)
    parser.add_argument('--max_episode_steps', help='', type=int, default=1000)
    parser.add_argument('--network', help='', type=str, default='cnn')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 2000', default=2000, type=int)
    parser.add_argument('--demo_path', help='Directory to save learning curve data.', default="data/demo/human.MontezumaRevengeNoFrameskip-v4.pkl", type=str)
    parser.add_argument('--log_path', help='Path to save log to', default='data/logs', type=str)
    parser.add_argument('--play', default=False, action='store_true')
    args = parser.parse_args()

    logger.configure(args.log_path)
    model, env = train(args)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()
        obs = np.expand_dims(np.array(obs), axis=0)

        state = model.initial_state if hasattr(model, 'initial_state') else None

        episode_rew = np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs)
            else:
              actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions.numpy())
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0
                    env.reset()
    env.close()

    return model


if __name__ == "__main__":
    main()
