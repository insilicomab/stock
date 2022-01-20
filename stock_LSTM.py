'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv', header=None)

# データの確認
print(train.head())
print(train.dtypes)

# 欠損値の確認
print(train.isnull().sum())
print(test.isnull().sum())

'''
特徴量エンジニアリング
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)

""" 特徴量Bodyを追加 """

df['Body'] = df['Open'] - df['Close']


""" 特徴量Rateを追加 """

df['Rate'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
df['Rate'] = df['Rate'].fillna(0)

""" 不要なカラムを削除 """

df = df.drop(['Date'], axis=1)

# trainとtestに再分割
train = df[~df['Up'].isnull()]
test = df[df['Up'].isnull()]

# 説明変数と目的変数を指定
X_train = train.drop(['Up'], axis=1)
Y_train = train['Up']
X_test = test.drop(['Up'], axis=1)
Y_test = test['Up']

# 学習データと検証データの分割
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=0.25, random_state=0,
                                                      shuffle=False)

# 関数get_standardized_tの定義
def get_standardized_t(X, num_date):
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date]
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_t)
        X_t_list.append(X_standardized)
    return np.array(X_t_list)

num_date = 5

# 関数get_standardized_tの呼び出し
x_train_t = get_standardized_t(X=x_train, num_date=num_date)
x_valid_t = get_standardized_t(X=x_valid, num_date=num_date)
X_test_t = get_standardized_t(X=X_test, num_date=num_date)

print(x_train_t.shape, x_valid_t.shape, X_test_t.shape)

# 目的変数の変形
y_train_t = y_train[num_date-1:]
y_valid_t = y_valid[num_date-1:]
Y_test_t = Y_test[num_date-1:]

'''
モデルの構築と評価
'''

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# ネットワークの各層のサイズの定義
num_l1 = 100
num_l2 = 20
num_output = 1

# エポック数
epochs = 10

# バッチサイズ
batch_size=24


# Dropoutの割合の定義
dropout_rate = 0.25

# 以下、ネットワークを構築
model = Sequential()
# 第1層
model.add(LSTM(units=num_l1, activation='tanh', batch_input_shape=(None, x_train_t.shape[1], x_train_t.shape[2])))
model.add(Dropout(dropout_rate))
# 第2層
model.add(Dense(num_l2, activation='relu'))
model.add(Dropout(dropout_rate))
# 出力層
model.add(Dense(num_output, activation='sigmoid'))
# ネットワークのコンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの学習の実行
result = model.fit(x=x_train_t, y=y_train_t, epochs=epochs, batch_size=batch_size, validation_data=(x_valid_t, y_valid_t))

# モデルを評価
score = model.evaluate(x_valid_t, y_valid_t, verbose=1)
print('正解率=', score[1], 'loss=', score[0])

'''
学習過程のグラフ化
'''
    
# 正解率の推移をプロット
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ロスの推移をプロット
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
テストデータの予測
"""

# 評価データの予測結果の算出
pred_prob = model.predict(X_test_t)

# 予測結果の先頭10件を確認
print('予測結果の先頭10件')
print(pred_prob[:10])

# 評価データの予測結果を0もしくは1に丸め込み
pred = np.round(pred_prob)

# 丸め込んだ予測結果の先頭10件を確認
print('丸め込んだ予測結果の先頭10件')
print(pred[:10], f'データ数：{len(pred)}')