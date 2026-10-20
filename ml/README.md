# drawing-classifier (ml)

MNISTの手書き数字を分類するPyTorchモデルの学習・ONNXエクスポート処理。

```bash
# 依存パッケージのインストール
uv sync

# 学習(data/ にMNISTを自動ダウンロード → models/mnist_model.pth に保存)
uv run python train.py

# ONNXエクスポート(models/mnist_model.pth → ../web/public/models/mnist_model.onnx)
uv run python export.py

# 再現性のある学習と検証データの分割(省略時は従来どおり)
uv run python train.py --seed 42 --validation-size 0.1 --epochs 5

# 単体テスト(MNISTのダウンロード不要。MPSのテストはApple siliconでのみ実行)
uv sync --group dev
uv run pytest
```

学習・評価・ONNXエクスポートの処理は `src/drawing_classifier/` の関数(`data`、`training`、`evaluation`、`onnx_export`)にあり、`train.py` と `export.py` はそれらを呼び出すCLIです。ONNXの入出力(`input`: [batch,1,28,28]、`output`: [batch,10])は変更しないでください。
