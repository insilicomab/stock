"""
コメント：
baseline作成前のEDA
"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime as dt

# 学習データの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format='%Y-%m-%d')

"""
可視化
"""

''' 全期間 '''

train_new = train.loc[:, ['Open', 'High', 'Low', 'Close']] # 特定のカラムの抽出
train_new.plot(kind='line') # 折れ線グラフの描画
plt.show()

''' Closeをプロット '''

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(train['Date'], train['Close'], label='Train')
ax.plot(test['Date'], test['Close'], label='Test')
plt.show()


''' Upの個数 '''

up_count = train['Up'].value_counts()
up_count.plot.bar() # 棒グラフ
plt.show()

print(up_count)

"""
特徴量エンジニアリング
"""

''' 始値-終値差分値 '''

train['Body'] = train['Open'] - train['Close']
test['Body'] = test['Open'] - test['Close']

