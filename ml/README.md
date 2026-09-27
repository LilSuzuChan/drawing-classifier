# drawing-classifier (ml)

MNISTの手書き数字を分類するPyTorchモデルの学習・ONNXエクスポート処理。

```bash
# 依存パッケージのインストール
uv sync

# 学習(data/ にMNISTを自動ダウンロード → models/mnist_model.pth に保存)
uv run python train.py

# ONNXエクスポート(models/mnist_model.pth → ../web/public/models/mnist_model.onnx)
uv run python export.py
```
