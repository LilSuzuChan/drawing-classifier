# Kubernetes マニフェスト

```sh
kubectl apply -k k8s/
```

## イメージ

`ghcr.io/lilsuzuchan/drawing-classifier` (非公開) は `.github/workflows/container.yml` が
push 時にビルドし、`sha-<push先頭コミットの短縮ID>` タグを付ける。`deployment.yaml` ではこのタグを指定する
(k8s/ だけの変更ではビルドされない)。

## pull 用 Secret (Git 管理外)

`read:packages` のみの GitHub PAT (classic) から `drawing-classifier` namespace に
`ghcr-pull` (kubernetes.io/dockerconfigjson) を作成する。トークンはファイルや履歴に残さない。

```sh
read -rsp 'PAT: ' T; echo
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}' "$(printf 'LilSuzuChan:%s' "$T" | base64 -w0)" \
  | kubectl -n drawing-classifier create secret generic ghcr-pull \
      --type=kubernetes.io/dockerconfigjson --from-file=.dockerconfigjson=/dev/stdin \
      --dry-run=client -o yaml \
  | kubectl apply --server-side --field-manager=ghcr-pull-script -f -
unset T
```
