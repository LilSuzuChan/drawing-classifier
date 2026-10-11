// logits(Softmax適用前の生の出力)を確率(合計1)へ変換する
export function softmax(logits: ArrayLike<number>): number[] {
  const maxLogit = Math.max(...Array.from(logits));

  // 最大値を引いてからexpすることで、大きな数値によるオーバーフローを防ぐ(数値安定化)
  const exps = Array.from(logits, (logit) => Math.exp(logit - maxLogit));
  const sumExps = exps.reduce((sum, value) => sum + value, 0);

  return exps.map((value) => value / sumExps);
}
