# フィードバックから自動再学習・モデル配信まで

画面のPredict後、正しい数字を選んで送信すると、推論時と同じ784個の0〜1画素、正解、予測値、モデル版をAPIへ保存します。描き直すと以前の予測への送信を無効化します。保存先は5Gi PVC上のSQLiteです。同一画像は1件だけ採用し、最大2000件で受付を止めます。上限到達時はDBの保全・データ整理が必要です。公開サービスで運用する場合は入口で利用者認証・レート制限も設定してください。

## 実行経路

`Snapshot → Train(MNIST + feedbackの約80%) → Evaluate → Export/ONNX parity → Compare → Promote`

残り約20%は画像ハッシュで固定した評価用です。追加データが増えても所属は変わりません。最低20件の評価用フィードバックが必要です。MNIST精度95%以上、現行モデル以上のMNIST精度、かつフィードバック評価精度の厳密な改善をすべて満たすと配信します。条件未達やデータ不足では現行モデルを維持し、理由をKFPのreportへ記録します。誤った正解の送信も学習に影響するため、運用で定期的に確認してください。

ONNX入出力は既存のままです。モデルと版を同じDBトランザクションで切り替え、同時実行で現行版が変わった場合は更新を拒否します。モデルを読み込んだブラウザはそのセッション中同じ版を使い、ページを開き直すと新しい版になります。過去のリリースはDBに保持します。APIの再起動でフィードバックと配信モデルは失われません。PVC喪失に備え、SQLiteのオンラインバックアップまたは停止中のPVCバックアップを別途設定してください。

## 初回導入

1. 変更をGitHubへpushし、`container.yml` と `ml-image.yml` を成功させます。Web/API/学習イメージのdigestを記録します。学習用の従来digestは新モジュールを含まないため使わないでください。
2. K8s-home-labをdrawing-classifierと同じ親ディレクトリに配置します。導入スクリプトが `platform/kubeflow-pipelines/app/networkpolicy.yaml` を反映します（kubeflowからfeedback-api:8080へのegress追加）。別配置の場合は `--lab-root` を指定します。
3. drawing-classifierとkubeflowの両Namespace、およびアプリNamespaceの `ghcr-pull` を用意します。API用PVCは既存クラスタの `local-path` StorageClassを明示して使用します。
4. `ml` で `uv sync --group dev --group pipeline` を実行します。
5. KFP UIをport-forwardし、次をリポジトリルートで実行します。

```bash
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80
# 別ターミナル。以下の各 <digest> はビルド成果物で置換
ml/.venv/bin/python ops/deploy.py \
  --web-image ghcr.io/lilsuzuchan/drawing-classifier@sha256:<digest> \
  --api-image ghcr.io/lilsuzuchan/drawing-classifier-feedback@sha256:<digest> \
  --training-image ghcr.io/lilsuzuchan/drawing-classifier-train@sha256:<digest> \
  --kfp-host http://127.0.0.1:8080
```

導入スクリプトは共有トークンSecretを両Namespaceへ作成し、既存トークンは再利用します。既存の `ghcr-pull` もKubeflowへ同期し、各KFPタスクのイメージ取得に使用します。SecretはGitへ書きません。両Deploymentの起動を待ってから、毎日03:00 JST（18:00 UTC）のKFP Recurring Runを登録します。並列数は1、取り逃した分は追実行しません。同名の旧スケジュールを無効化して重複実行を防ぎます。`pipeline-runner` はKFPの既存SAを使います。

APIの内部パスはトークン必須で、nginxからは公開しません。Namespace間通信はNetworkPolicyでAPIの8080番だけを許可します。ネットワーク上のHTTPなので、信頼できるホームラボ内を前提としています。

## 検証

```bash
cd ml
uv run pytest
uv run python -m pipeline.training_pipeline
uv run python -m pipeline.continuous_pipeline
# 実際の登録時は新イメージdigestを指定する
DRAWING_TRAINING_IMAGE=ghcr.io/lilsuzuchan/drawing-classifier-train@sha256:<digest> \
  uv run python -m pipeline.schedule --host http://127.0.0.1:8080
```

`drawing-classifier-continuous.yaml` は生成物です。登録スクリプトは指定した新digestで再コンパイルします。ソース内の従来digestによるローカルコンパイル結果をそのままクラスタへ登録しないでください。

学習コンテナのテストには追加のAPIコンテキストが必要です。

```bash
docker buildx build ml --target test-report \
  --build-context web-models=web/public/models --build-context api-sources=api \
  --output type=local,dest=pytest-report
```

実環境では、画面送信→Pod再起動後の保持→KFP Snapshotの取得→データ不足時に未配信→十分なデータでの評価レポートを確認します。サンプルを捏造して本番DBへ入れる必要はありません。

## ローカル画面開発

リポジトリルートでAPIを起動し、別ターミナルでwebの開発サーバーを起動します。Viteの/apiはlocalhost:8080へ転送します。

```bash
PYTHONPATH=ml/src FEEDBACK_DB=/tmp/drawing-feedback-dev.sqlite3 \
  FEEDBACK_TOKEN=local-development-token-at-least-32-characters \
  BUNDLED_MODEL=web/public/models/mnist_model.onnx ml/.venv/bin/python api/server.py
cd web && npm ci && npm run dev
```
