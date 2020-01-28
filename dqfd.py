import os.path as osp
from tqdm import tqdm
from time import time
from collections import deque
import pickle
import tensorflow as tf
import numpy as np

from common.schedules import LinearSchedule, ConstantSchedule
from common.misc_util import set_global_seeds, timedelta
from common import logger

from replay_buffer import PrioritizedReplayBuffer

from models import build_q_func
from dqfd_learner import DQfD


def get_n_step_sample(buffer, gamma):
    """
    N ステップ分の割引報酬和を計算してから行動軌跡をリプレイバッファに保存するための関数です。
    割引報酬和は前のステップの計算結果から再帰的に計算するやりかたの方が速そうなのですが、誤差が増幅してうまく行かなかったので普通に計算しています。
    """
    reward_n = 0
    for i, step in enumerate(buffer):
        reward_n += step[2] * (gamma ** i)
    obs     = buffer[0][0]
    action  = buffer[0][1]
    rew     = buffer[0][2]
    new_obs = buffer[0][3]
    done    = buffer[0][4]
    is_demo = buffer[0][5]
    n_step_obs = buffer[-1][3]
    done_n  = buffer[-1][4]
    return obs[0], action, rew, new_obs[0], float(done), float(is_demo), n_step_obs[0], reward_n, done_n


def learn(env,
          network,
          seed=None,
          lr=1e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=1,
          batch_size=32,
          print_freq=10000,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=1000,
          prioritized_replay=True,
          prioritized_replay_alpha=0.4,
          prioritized_replay_beta0=0.6,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-3,
          param_noise=False,
          callback=None,
          load_path=None,
          load_idx=None,
          # dqfd
          demo_path=None,
          n_step=10,
          demo_prioritized_replay_eps=1.,
          pre_train_timesteps=750000,
          epsilon_schedule="constant",
          **network_kwargs
            ):
    """dqfdのモデルを学習します.

    Parameters
    -------
    env: gyn.Env
        学習する環境
    network: string or a function
        関数近似に用いるニューラルネット。stringの場合はcommon.modelsに登録されたモデル(mlp, cnn, conv_only)。
    seed: int or None
        同じシード値は同じ学習結果を与える「はず」です。
    lr: float
        Adam optimizerの学習率
    total_timesteps: int
        学習する環境の総ステップ数(事前学習のステップ数は含みません)
    buffer_size: int
        リプレイバッファのサイズ
    exploration_fraction: float
        ε-greedyのεを減少させる期間(dqfdの場合はεは0.01に固定)
    exploration_final_eps: float
        ε-greedyのεの最終値
    train_freq: int
        学習を行う頻度
    batch_size: int
        バッチサイズ
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    checkpoint_path: str
        path to save the model to.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    demo_path: str
        path to load the demo from. (default: None)
    n_step: int
        number of steps for calculating N step TD error
    demo_prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities of demo.
    pre_train_timesteps: int
        number of pre-training steps to using demo
    epsilon_schedule:str
        epsilon-greedy schedule (default:constant)
    **network_kwargs
        additional keyword arguments to pass to the network builder.
    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    set_global_seeds(seed)
    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    model = DQfD(
        q_func=q_func,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=lr,
        grad_norm_clipping=10,
        gamma=gamma,
        param_noise=param_noise
    )

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        if load_idx is None:
            ckpt.restore(manager.latest_checkpoint)
            print("Restoring from {}".format(manager.latest_checkpoint))
        else:
            ckpt.restore(manager.checkpoints[load_idx])
            print("Restoring from {}".format(manager.checkpoints[load_idx]))

    # Setup demo trajectory
    assert demo_path is not None
    with open(demo_path, "rb") as f:
        trajectories = pickle.load(f)
    # Create the replay buffer
    replay_buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = total_timesteps
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                   initial_p=prioritized_replay_beta0,
                                   final_p=1.0)
    temp_buffer = deque(maxlen=n_step)
    is_demo = True
    for epi in trajectories:
        for obs, action, rew, new_obs, done in epi:
            obs, new_obs = np.expand_dims(np.array(obs), axis=0), np.expand_dims(np.array(new_obs), axis=0)
            if n_step:
                temp_buffer.append((obs, action, rew, new_obs, done, is_demo))
                if len(temp_buffer) == n_step:
                    n_step_sample = get_n_step_sample(temp_buffer, gamma)
                    replay_buffer.demo_len += 1
                    replay_buffer.add(*n_step_sample)
            else:
                replay_buffer.demo_len += 1
                replay_buffer.add(obs[0], action, rew, new_obs[0], float(done), float(is_demo))
    # Create the schedule for exploration
    if epsilon_schedule == "constant":
        exploration = ConstantSchedule(exploration_final_eps)
    else:
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

    model.update_target()

    # ============================================== pre-training ======================================================
    start = time()
    temp_buffer = deque(maxlen=n_step)  # reset temporary buffer
    for t in tqdm(range(pre_train_timesteps)):
        # sample and train
        experience = replay_buffer.sample(batch_size, beta=prioritized_replay_beta0)
        (obses_t, actions, rewards, obses_tp1, dones, is_demos, obses_tpn, rewards_n, dones_n, weights, batch_idxes) = experience
        obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
        actions, rewards, dones, is_demos = tf.constant(actions), tf.constant(rewards), tf.constant(dones), tf.constant(is_demos)
        weights = tf.constant(weights)
        if obses_tpn is not None:
            obses_tpn, rewards_n, dones_n = tf.constant(obses_tpn), tf.constant(rewards_n), tf.constant(dones_n)
        td_errors, n_td_errors, loss_dq, loss_n, loss_E, loss_l2, weighted_error = model.train(obses_t, actions, rewards, obses_tp1, dones, is_demos, weights, obses_tpn, rewards_n, dones_n)

        # update priorities
        new_priorities = np.abs(td_errors) + np.abs(n_td_errors) + demo_prioritized_replay_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Update target network periodically.
        if t > 0 and t % target_network_update_freq == 0:
            model.update_target()

        # logging
        num_episodes = 0
        elapsed_time = timedelta(time() - start)
        if print_freq is not None and t % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", 0)
            logger.record_tabular("max 100 episode reward", 0)
            logger.record_tabular("min 100 episode reward", 0)
            logger.record_tabular("demo sample rate", 1)
            logger.record_tabular("epsilon", 0)
            logger.record_tabular("loss_td", np.mean(loss_dq.numpy()))
            logger.record_tabular("loss_n_td", np.mean(loss_n.numpy()))
            logger.record_tabular("loss_margin", np.mean(loss_E.numpy()))
            logger.record_tabular("loss_l2", np.mean(loss_l2.numpy()))
            logger.record_tabular("losses_all", weighted_error.numpy())
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("pre_train", True)
            logger.record_tabular("elapsed time", elapsed_time)
            logger.dump_tabular()

    # ============================================== exploring =========================================================
    sample_counts = 0
    demo_used_counts = 0
    episode_rewards = deque(maxlen=100)
    this_episode_reward = 0.
    saved_mean_reward = None
    is_demo = False
    obs = env.reset()
    # always mimic the vectorized env
    obs = np.expand_dims(np.array(obs), axis=0)
    reset = True
    for t in tqdm(range(total_timesteps)):
        if callback is not None:
            if callback(locals(), globals()):
                break
        kwargs = {}
        if not param_noise:
            update_eps = tf.constant(exploration.value(t))
            update_param_noise_threshold = 0.
        else:
            update_eps = tf.constant(0.)
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action, epsilon, _, _ = model.step(tf.constant(obs), update_eps=update_eps, **kwargs)
        action = action[0].numpy()
        reset = False
        new_obs, rew, done, _ = env.step(action)

        # Store transition in the replay buffer.
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        if n_step:
            temp_buffer.append((obs, action, rew, new_obs, done, is_demo))
            if len(temp_buffer) == n_step:
                n_step_sample = get_n_step_sample(temp_buffer, gamma)
                replay_buffer.add(*n_step_sample)
        else:
            replay_buffer.add(obs[0], action, rew, new_obs[0], float(done), 0.)
        obs = new_obs

        this_episode_reward += np.sign(rew) * (np.exp(np.sign(rew) * rew) - 1.)  # 記録用にlogスケールを元に戻す
        if done:
            obs = env.reset()
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rewards.append(this_episode_reward)
            reset = True
            this_episode_reward = 0.

        if t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.=============
            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
            (obses_t, actions, rewards, obses_tp1, dones, is_demos, obses_tpn, rewards_n, dones_n, weights, batch_idxes) = experience
            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones, is_demos = tf.constant(actions), tf.constant(rewards), tf.constant(dones), tf.constant(is_demos)
            weights = tf.constant(weights)
            if obses_tpn is not None:
                obses_tpn, rewards_n, dones_n = tf.constant(obses_tpn), tf.constant(rewards_n), tf.constant(dones_n)
            td_errors, n_td_errors, loss_dq, loss_n, loss_E, loss_l2, weighted_error = model.train(obses_t, actions, rewards, obses_tp1,
                                                                              dones, is_demos, weights, obses_tpn,
                                                                              rewards_n, dones_n)
            new_priorities = np.abs(td_errors) + np.abs(n_td_errors) + demo_prioritized_replay_eps * is_demos + prioritized_replay_eps * (1. - is_demos)
            replay_buffer.update_priorities(batch_idxes, new_priorities)

            # ログ用
            sample_counts += batch_size
            demo_used_counts += np.sum(is_demos)

        if t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

        if t % checkpoint_freq == 0:
            save_path = checkpoint_path
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=10)
            manager.save(t)
            logger.log("saved checkpoint")

        num_episodes = len(episode_rewards)
        elapsed_time = timedelta(time() - start)
        if done and print_freq is not None and len(episode_rewards) > 0 and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", np.mean(episode_rewards))
            logger.record_tabular("max 100 episode reward", np.max(episode_rewards))
            logger.record_tabular("min 100 episode reward", np.min(episode_rewards))
            logger.record_tabular("demo sample rate", demo_used_counts / sample_counts)
            logger.record_tabular("epsilon", epsilon.numpy())
            logger.record_tabular("loss_td", np.mean(loss_dq.numpy()))
            logger.record_tabular("loss_n_td", np.mean(loss_n.numpy()))
            logger.record_tabular("loss_margin", np.mean(loss_E.numpy()))
            logger.record_tabular("loss_l2", np.mean(loss_l2.numpy()))
            logger.record_tabular("losses_all", weighted_error.numpy())
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("pre_train", False)
            logger.record_tabular("elapsed time", elapsed_time)
            logger.dump_tabular()
    return model





