# Replication of DQfD(Deep Q-Learning from Demonstrations)
This repo replicates the results Hester et al obtained:
[Deep Q-Learning from Demonstraitions](https://arxiv.org/abs/1704.03732 "Deep Q-Learning from Demonstraitions")

This code is based off of code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines "here").

このリポジトリはHesterらによるDeep Q-Learning from Demonstrations(DQfD)を再現実装したものです。

このコードはOpenAI baselinesに基づいて実装されています。

オリジナルのコードやそれに関連する論文については[こちら](https://github.com/openai/baselines "こちら").
を参照してください。

# 使い方
clone git
```python:
git clone 
```

create conda virtual env and activate
```python:
conda create -n venv
conda activate venv
```

install requirements
```python:
pip install tensorflow=2.0
pip install tqdm
pip install gym
pip install gym[atari]
```

make demonstrations
```python:
python make_demo.py --env=MontezumaRevengeNoFrameskip-v4
```

run training
```python:
python run_atari.py
```
