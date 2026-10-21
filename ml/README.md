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

## 学習用コンテナイメージ

`Dockerfile` はKFPの学習パイプライン用のイメージ(`ghcr.io/lilsuzuchan/drawing-classifier-train`)を作ります。Linuxでは `pyproject.toml` の設定によりCPU版のtorchを使い、`pipeline` グループの `kfp` を含みます。MNISTはビルド時に `container/mnist.sha256` で検証してから `/opt/mnist` に格納するため、実行時にネットワークは不要です(`load_mnist("/opt/mnist", download=False)`)。実行ユーザーはUID/GID 1000です。

```bash
# pipelineグループも含めた依存パッケージのインストール
uv sync --group dev --group pipeline

# イメージ内のテスト(testステージ)とイメージのビルド。リポジトリのルートで実行
# testステージはpytestの結果をファイルに残すため、終了コードは exit-code で確認する
docker buildx build ml --target test-report --build-context web-models=web/public/models --output type=local,dest=pytest-report
cat pytest-report/output.txt && test "$(cat pytest-report/exit-code)" = 0
docker buildx build ml --target runtime -t drawing-classifier-train
```

`.github/workflows/ml-image.yml` が、testステージの成功後にイメージをGHCRへpushします。パイプラインからはタグではなくdigestで参照してください。

## KFP学習パイプライン

`pipeline/` はKFP 2.17向けの学習パイプライン(`drawing-classifier-training`)です。Train → Evaluate → Export の3ステップを、digestで固定した学習用イメージで順に実行します。各ステップは `src/drawing_classifier/` の関数を呼び出すだけで、学習処理は持ちません。

| ステップ | 入力 | 出力 |
|---|---|---|
| Train | 学習条件、イメージ内のMNIST | `model`(state_dict)、`training_config`(学習条件・イメージ・データセットのSHA256)、`metrics` |
| Evaluate | `model`、`training_config` | `metrics`(test_accuracy、test_loss)、`confusion_matrix`、`evaluation`、合格判定(bool) |
| Export | `model`、`evaluation` | `onnx_model`(Webアプリと同じ入出力)、`metrics`(PyTorchとの差) |

```bash
# pipelineグループを含めて依存パッケージをインストール
uv sync --group dev --group pipeline

# パイプラインをコンパイル(pipeline/drawing-classifier-training.yaml を更新)
uv run python -m pipeline.training_pipeline

# KFPへ登録して実行(kubectlでml-pipeline-uiへport-forwardする)
uv run python -m pipeline.run --epochs 1
```

`pipeline/run.py` は、同じExperimentに実行中のrunがあると新しいrunを作りません。ServiceAccountは `pipeline-runner`、キャッシュは無効です。
