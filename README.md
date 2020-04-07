# Replication of DQfD(Deep Q-Learning from Demonstrations)
This repo replicates the results Hester et al. obtained:
[Deep Q-Learning from Demonstraitions](https://arxiv.org/abs/1704.03732 "Deep Q-Learning from Demonstraitions")  
This code is based on code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines "here").  
<br/>
このリポジトリはHesterらによるDeep Q-Learning from Demonstrations(DQfD)を再現実装したものです。  
アルゴリズムやハイパーパラメータなどはできる限り論文をもとにしていますが、完全に同じパフォーマンスを再現することはできません。  
<br/>
このコードはOpenAI baselinesに基づいて実装されています。  
オリジナルのコードやそれに関連する論文については[こちら](https://github.com/openai/baselines "こちら")を参照してください。  
このアルゴリズムに関するブログはこちら[ブログのURL]を参照してください。  
<br/>
# 環境のセットアップについて
必要なライブラリは  
・Tensorflow2(GPUを使用する場合tensorflow-gpu)  
・gym  
・gym[atari]  
・tqdm  
です。  
(GPUを使用しない場合はdqfd.pyの71行目  
with tf.device('/GPU:0'):  
を  
with tf.device('/CPU:0'):  
と書き換えてください。)  
<br/>
## Ubuntu 18.04でのセットアップ例
リポジトリをクローン
```python:
git clone https://github.com/morikatron/DQfD.git
```

仮想環境を作成してアクティベート
```python:
conda create -n DQfDenv
conda activate DQfDenv
```

必要なライブラリをインストール
```python:
conda install tensorflow-2.0
(conda install tensorflow-gpu)

pip install gym
pip install gym[atari]
(エラーが出る場合は pip install 'gym[atari]')
pip install tqdm
```


# 使い方
まずmake_demo.pyを実行してデモを作成します。  
作成したデモは./data/demoディレクトリに保存されます。  
例
```python:
python make_demo.py --env=MontezumaRevengeNoFrameskip-v4
```
### 操作方法  
・w,s,a,d：上下左右に移動  
・SPACE：ジャンプ  
・backspace：このエピソードの行動軌跡を保存せずリセット  
・return：このエピソードの行動軌跡を保存してリセット  
・esc：このエピソードの行動軌跡を保存せずゲームを終了  
<br/>
デモの作成が完了したらrun_atari.pyを実行して学習を開始します。  
```python:
python run_atari.py
```
### run_atari.pyの引数  
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

# デモデータ
Montezuma's Revengeでステージ1をクリアした5エピソード分のデモデータを以下のリンク先に置いておきます。(サイズが906MBと大きいので注意です)  
https://drive.google.com/file/d/1bxfIkqxjiJKH9Pg2a8ZRMIheX7wypEJL/view?usp=sharing  
リンク先のpklファイルをDQfD/data/demoディレクトリに配置することでデモを作成せずに学習を開始することができます。

# Mac OSでエラーが出る場合
Mac OSでOMP: Errorが出る場合、dqfd.pyの頭に  
```python:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
```
を加えてください。(他の解決方法をご存じであれば教えていただけるとありがたいです。)
