# Replication of DQfD(Deep Q-Learning from Demonstrations)
This repo replicates the results Hester et al. obtained:
[Deep Q-Learning from Demonstraitions](https://arxiv.org/abs/1704.03732 "Deep Q-Learning from Demonstraitions")

This code is based on code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines "here").

このリポジトリはHesterらによるDeep Q-Learning from Demonstrations(DQfD)を再現実装したものです。

アルゴリズムやハイパーパラメータなどはできる限り論文をもとにしていますが、完全に同じパフォーマンスを再現することはできません。

このコードはOpenAI baselinesに基づいて実装されています。

オリジナルのコードやそれに関連する論文については[こちら](https://github.com/openai/baselines "こちら")
を参照してください。

このアルゴリズムに関するブログは[こちら]( "こちら")を参照してください。

# 環境のセットアップについて
必要なライブラリは

・Tensorflow2(GPUを使用する場合tensorflow-gpu)

・gym

・gym[atari]

・tqdm

です。

(GPUを使用しない場合はdqfd.pyの71行目を

with tf.device('/GPU:0'): -> with tf.device('/CPU:0'):

に書き換えてください。)

## Mac OS Xでのセットアップ例
clone git
```python:
git clone https://github.com/morikatron/DQfD.git
```

create conda virtual env and activate
```python:
conda create -n DQfDenv
conda activate DQfDenv
```

install requirements
```python:
conda install tensorflow-2.0
(conda install tensorflow-gpu)

pip install gym
pip install gym[atari]
pip install tqdm
```

# 使い方
まずmake_demo.pyを実行してデモを作成します。
作成したデモは./data/demoディレクトリに保存されます。
例
```python:
python make_demo.py --env=MontezumaRevengeNoFrameskip-v4
```

デモの作成が完了したらrun_atari.pyを実行して学習を開始します。
```python:
python run_atari.py
```
