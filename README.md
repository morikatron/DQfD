# Replication of DQfD(Deep Q-Learning from Demonstrations)
![montezuma_result](https://github.com/morikatron/DQfD/blob/master/assets/montezuma.gif "montezuma_result")
This repo replicates the results Hester et al. obtained:
[Deep Q-Learning from Demonstraitions](https://arxiv.org/abs/1704.03732 "Deep Q-Learning from Demonstraitions")  
This code is based on code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines "here").  
<br/>
このリポジトリはHesterらによるDeep Q-Learning from Demonstrations(DQfD)を再現実装したものです。  
アルゴリズムやハイパーパラメータなどはできる限り論文をもとにしていますが、完全に同じパフォーマンスを再現することはできません。  
<br/>
このコードはOpenAI baselinesに基づいて実装されています。  
オリジナルのコードやそれに関連する論文については[こちら](https://github.com/openai/baselines "こちら")を参照してください。  
このアルゴリズムに関するブログは[こちら](https://tech.morikatron.ai/entry/2020/04/15/100000)を参照してください。  
<br/>
## Requirements
・Python3
・tensorflow2(or tensorflow-gpu)  
・gym  
・gym[atari]  
・tqdm  
(when you don't use gpu, please rewrite line 71 on dqfd.py  
with tf.device('/GPU:0'):  
to  
with tf.device('/CPU:0'):  
)  
<br/>
### Usage(Ubuntu 18.04)
 - clone this repo
```python:
git clone https://github.com/morikatron/DQfD.git
```
<br/>

 - activate virtual environment(if needed)  
 
conda  
```python:
conda create -n DQfDenv
conda activate DQfDenv
```
venv
```python:
python3 -m venv DQfDenv
source DQfDenv/bin/activate
```
<br/>

 - install packages
(example for conda environment)
```python:
conda install tensorflow-2.0
(conda install tensorflow-gpu)

pip install gym
pip install gym[atari]
(If you get an error, try pip install 'gym[atari]')
pip install tqdm
```
<br/>

 - create demo
Demo trajectories are needed to train agent.  
(example for Montezuma's Revenge demo)
```python:
python make_demo.py --env=MontezumaRevengeNoFrameskip-v4
```
Created trajectories are saved to ./data/demo directory.  
<br/>

 - Start training
```python:
python run_atari.py
```
<br/>

### how to operate a demo
・w,s,a,d：move up, down, left, right  
・SPACE：jump  
・backspace：reset this episode without saving this episode's trajectory  
・return：reset this episode saving this episode's trajectory  
・esc：quit game without saving this episode's trajectory  
・plus: speed up(doubles fps)  
・minus: speed down(halves fps)
<br/>

### command-line options of run_atari.py  
コマンドライン引数で学習時の設定を指定することができます。  
・env : 学習を行う環境(デフォルトはMontezumaRevengeNoFrameskip-v4 必ずデモの環境と同じものを指定してください)  
・pre_train_timesteps：事前学習を行うステップ数(デフォルトは75万)  
・num_timesteps：(事前学習を除く)学習を行う総ステップ数(デフォルトは100万)  
・demo_path：デモデータが保存してあるパス  
・play：学習後にプレイを実行  
例
```python:
python run_atari.py  --pre_train_timesteps=1e6 --num_timesteps=1e7 
```
他のパラメータについてはrun_atari.pyのmain()関数をご確認ください。

## デモデータ
Montezuma's Revengeでステージ1をクリアした5エピソード分のデモデータを以下のリンク先に置いておきます。(サイズが906MBと大きいので注意です)  
https://drive.google.com/file/d/1bxfIkqxjiJKH9Pg2a8ZRMIheX7wypEJL/view?usp=sharing  
リンク先のpklファイルをDQfD/data/demoディレクトリに配置することでデモを作成せずに学習を開始することができます。

## Mac OSでエラーが出る場合
Mac OSでOMP: Errorが出る場合、dqfd.pyの頭に  
```python:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
```
を加えてください。(他の解決方法をご存じであれば教えていただけるとありがたいです。)
