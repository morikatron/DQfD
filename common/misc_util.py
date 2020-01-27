import gym
import numpy as np
import os
import pickle
import random
import tempfile
import zipfile


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def timedelta(seconds):
    """
    reference: https://qiita.com/mitama/items/7726ff2ecd80f3b10648
    秒数が1秒未満の場合はミリ秒を、1秒以上の場合はhh:mm:ssの形式に変換する"""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    else:
        m, s = divmod(round(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"