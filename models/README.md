# モデルディレクトリ

この `models` ディレクトリは、機械学習モデルの定義、トレーニング、評価、そして予測を行うためのコードを含んでいます。

## ディレクトリ構造

- `cnn_model.py` - 畳み込みニューラルネットワーク（CNN）モデルの定義が含まれています。
- `train_model.py` - モデルのトレーニングを行うスクリプトです。モデルの保存もここで行われます。
- `evaluate_model.py` - トレーニング済みモデルの評価と新しいデータに対する予測を行うスクリプトです。

## 使用方法

### モデルの定義
`cnn_model.py` を使用して、任意の入力形状とクラス数に基づいてCNNモデルを生成できます。使用例は以下の通りです。

```python
from models.cnn_model import create_cnn_model
model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)
```

### モデルのトレーニング
`train_model.py` を実行することで、モデルをトレーニングし、指定したパスに保存することができます。例を以下に示します。

```python
from models.train_model import train_model
model = train_model(data_train, data_test, input_shape=(28, 28, 1), num_classes=10)
```

### モデルの評価と予測
`evaluate_model.py` でモデルをロードし、テストデータセットで評価を行ったり、新しいデータに対して予測を行うことができます。

```python
from models.evaluate_model import load_model, evaluate_model, predict
model = load_model('path/to/save/model.h5')
results = evaluate_model(model, data_test)
predictions = predict(model, new_data)
```

## 注意事項
- モデルのトレーニングや評価には大量のデータと計算リソースが必要です。適切な環境でスクリプトを実行してください。